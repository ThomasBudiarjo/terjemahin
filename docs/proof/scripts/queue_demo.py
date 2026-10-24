"""Queue round trip: Go API (POST /v1/pages) -> RabbitMQ -> `terjemahin-worker serve` -> results -> API.

Also publishes one malformed and one direct JobMessage straight to the queue with pika.
usage: queue_demo.py API_URL PAGE STORAGE_DIR
"""

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pika

api, page, storage = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http(method, url, body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


# 1) through the Go API
boundary = uuid.uuid4().hex
parts = []
for name, val in [("pipeline", json.dumps({"inpainter": "lama_manga"})), ("debug", "true")]:
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
parts.append(
    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{page.name}"\r\n'
    f"Content-Type: image/png\r\n\r\n".encode() + page.read_bytes() + b"\r\n"
)
parts.append(f"--{boundary}--\r\n".encode())
t0 = time.time()
st, _, body = http("POST", api + "/v1/pages", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
job = json.loads(body)
log("POST /v1/pages ->", st, job)
jid = job.get("job_id") or job.get("id")
while True:
    st, _, body = http("GET", f"{api}/v1/jobs/{jid}")
    j = json.loads(body)
    if j.get("status") not in ("queued", "running", "pending"):
        break
    time.sleep(1)
log(f"GET /v1/jobs/{jid} ->", st, j, f"after {time.time() - t0:.1f}s")
st, ct, img = http("GET", f"{api}/v1/jobs/{jid}/image")
Path(storage, "api_result.png").write_bytes(img)
log("GET image ->", st, ct, len(img), "bytes")
st, ct, res = http("GET", f"{api}/v1/jobs/{jid}/result")
r = json.loads(res)
log("GET result ->", st, ct, len(r["bubbles"]), "bubbles; first:", r["bubbles"][0]["source_text"], "->", r["bubbles"][0]["target_text"])
for stage in ("detect", "mask", "inpainted", "final"):
    try:
        st, ct, b = http("GET", f"{api}/v1/jobs/{jid}/debug/{stage}")
        log(f"GET debug/{stage} ->", st, ct, len(b), "bytes")
    except Exception as e:  # noqa: BLE001
        log(f"GET debug/{stage} ->", e)

# 2) malformed message straight onto the queue: the worker must answer 'failed', not hang
conn = pika.BlockingConnection(pika.URLParameters("amqp://guest:guest@localhost:5672/"))
ch = conn.channel()
ch.queue_declare("terjemahin.jobs", durable=True)
bad = {"job_id": "bad-" + uuid.uuid4().hex[:6], "page_id": "p", "input_path": 42}
ch.basic_publish("", "terjemahin.jobs", json.dumps(bad).encode())
log("published malformed job", bad["job_id"])
conn.close()
time.sleep(3)
try:
    st, _, body = http("GET", f"{api}/v1/jobs/{bad['job_id']}")
    log("API view of malformed job ->", st, body.decode()[:300])
except Exception as e:  # noqa: BLE001
    log("API view of malformed job ->", e, "(job was never created through the API)")
