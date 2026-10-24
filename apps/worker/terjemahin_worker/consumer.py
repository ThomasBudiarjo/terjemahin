from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from .config import Settings
from .imageio import read_image
from .models import JobMessage
from .pipeline import Pipeline

log = logging.getLogger(__name__)

RESULT_QUEUE_SUFFIX = ".results"


def _resolve(storage_dir: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else storage_dir / path


def bad_message_status(body: bytes, err: Exception) -> dict | None:
    """A 'failed' result for a message that does not validate but names its job, or None."""
    try:
        raw = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("job_id"), str):
        return None
    msg = str(err).splitlines()
    return {
        "job_id": raw["job_id"],
        "page_id": str(raw.get("page_id", "")),
        "status": "failed",
        "error": "invalid job message: " + " ".join(m.strip() for m in msg[:6]),
    }


def process(job: JobMessage, pipeline: Pipeline, settings: Settings) -> dict:
    """Run one job and write its outputs; the status dict that goes to the results queue."""
    status: dict = {"job_id": job.job_id, "page_id": job.page_id}
    try:
        page = read_image(_resolve(settings.storage_dir, job.input_path), settings.max_pixels)
        res = pipeline.run(page, job.context, job.pipeline, debug=bool(job.debug_dir))
        out_img, result = res.image, res.result
        if job.debug_dir:
            dbg = _resolve(settings.storage_dir, job.debug_dir)
            dbg.mkdir(parents=True, exist_ok=True)
            for name, img in res.debug.items():
                cv2.imwrite(str(dbg / f"{name}.png"), img)

        out_path = _resolve(settings.storage_dir, job.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), out_img):
            raise OSError(f"cannot write {out_path}")
        result_path = _resolve(settings.storage_dir, job.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(result.model_dump_json())
        status.update(status="done", bubbles=len(result.bubbles))
    except Exception as e:
        log.exception("job %s failed", job.job_id)
        status.update(status="failed", error=f"{type(e).__name__}: {e}")
    return status


def run(settings: Settings) -> None:
    pipeline = Pipeline.from_settings(settings)
    params = pika.URLParameters(settings.amqp_url)
    params.heartbeat = 600  # one page can take a minute on CPU; keep the broker patient
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(settings.amqp_queue, durable=True)
    ch.queue_declare(settings.amqp_queue + RESULT_QUEUE_SUFFIX, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_message(
        channel: BlockingChannel, method: Basic.Deliver, _props: BasicProperties, body: bytes
    ) -> None:
        try:
            job = JobMessage.model_validate_json(body)
        except Exception as e:
            log.exception("bad message, dropping: %r", body[:200])
            failed = bad_message_status(body, e)
            if failed is not None:  # the API would otherwise show the job as queued forever
                channel.basic_publish(
                    exchange="",
                    routing_key=settings.amqp_queue + RESULT_QUEUE_SUFFIX,
                    body=json.dumps(failed).encode(),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
            channel.basic_nack(method.delivery_tag, requeue=False)
            return

        status = process(job, pipeline, settings)
        channel.basic_publish(
            exchange="",
            routing_key=settings.amqp_queue + RESULT_QUEUE_SUFFIX,
            body=json.dumps(status).encode(),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        channel.basic_ack(method.delivery_tag)

    ch.basic_consume(settings.amqp_queue, on_message)
    log.info("consuming %s", settings.amqp_queue)
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        ch.stop_consuming()
    finally:
        conn.close()
