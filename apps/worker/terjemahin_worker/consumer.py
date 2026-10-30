"""`terjemahin-worker serve`: take jobs from the web app (apps/web) over HTTP and run them.

The worker never touches the queue, database or bucket itself, so it runs anywhere that can
reach the app (no inbound port needed):

    POST {api_url}/api/worker/claim               -> 204, or {lease, job: JobMessage}
    GET  {api_url}/api/worker/jobs/{id}/input     -> the page
    POST {api_url}/api/worker/jobs/{id}/result    <- lease, status=done|failed, files
"""

from __future__ import annotations

import logging
import time

import cv2
import httpx

from .config import Settings
from .imageio import decode_image
from .models import JobMessage
from .pipeline import Pipeline

log = logging.getLogger(__name__)


def process(job: JobMessage, data: bytes, pipeline: Pipeline, settings: Settings) -> dict:
    """Run one job: the multipart fields and files for its result post."""
    try:
        page = decode_image(data, settings.max_pixels)
        res = pipeline.run(page, job.context, job.pipeline, debug=job.debug)
        ok, png = cv2.imencode(".png", res.image)
        if not ok:
            raise OSError("cannot encode the output page")
        files = {
            "output": ("output.png", png.tobytes(), "image/png"),
            "result": ("result.json", res.result.model_dump_json().encode(), "application/json"),
        }
        for name, img in res.debug.items() if job.debug else ():
            files[f"debug_{name}"] = (f"{name}.png", cv2.imencode(".png", img)[1].tobytes(),
                                      "image/png")
        return {"data": {"status": "done", "bubbles": str(len(res.result.bubbles))},
                "files": files}
    except Exception as e:
        log.exception("job %s failed", job.job_id)
        return {"data": {"status": "failed", "error": f"{type(e).__name__}: {e}"}, "files": {}}


def run_once(client: httpx.Client, pipeline: Pipeline, settings: Settings) -> bool:
    """Claim and run one job; False when there was nothing to do."""
    r = client.post("/api/worker/claim")
    r.raise_for_status()
    if r.status_code == 204:
        return False
    claim = r.json()
    job_id = str(claim["job"].get("job_id", ""))
    try:
        job = JobMessage.model_validate(claim["job"])
    except Exception as e:  # the app would otherwise show the job as running until it expires
        log.exception("bad job message %r", claim["job"])
        out = {"data": {"status": "failed", "error": f"invalid job message: {e}"}, "files": {}}
    else:
        inp = client.get(f"/api/worker/jobs/{job_id}/input")
        inp.raise_for_status()
        out = process(job, inp.content, pipeline, settings)
    data = {"lease": claim["lease"], **out["data"]}
    client.post(f"/api/worker/jobs/{job_id}/result", data=data,
                files=out["files"] or None).raise_for_status()
    log.info("job %s %s", job_id, data["status"])
    return True


def run(settings: Settings) -> None:
    if not settings.worker_token:
        raise SystemExit("WORKER_TOKEN is not set (the app's WORKER_TOKEN secret)")
    pipeline = Pipeline.from_settings(settings)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per poll otherwise
    client = httpx.Client(base_url=settings.api_url.rstrip("/"), timeout=120,
                          headers={"authorization": f"Bearer {settings.worker_token}"})
    log.info("taking jobs from %s", settings.api_url)
    try:
        while True:
            try:
                if run_once(client, pipeline, settings):
                    continue
            except httpx.HTTPError as e:  # app down or deploying: keep trying
                log.warning("api: %s", e)
            time.sleep(settings.poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
