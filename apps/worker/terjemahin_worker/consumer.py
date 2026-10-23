from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from .config import Settings
from .models import JobMessage
from .pipeline import Pipeline

log = logging.getLogger(__name__)

RESULT_QUEUE_SUFFIX = ".results"


def _resolve(storage_dir: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else storage_dir / path


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
        except Exception:
            log.exception("bad message, dropping: %r", body[:200])
            channel.basic_nack(method.delivery_tag, requeue=False)
            return

        status: dict = {"job_id": job.job_id, "page_id": job.page_id}
        try:
            page = cv2.imread(str(_resolve(settings.storage_dir, job.input_path)))
            if page is None:
                raise FileNotFoundError(job.input_path)
            res = pipeline.run(page, job.context, job.pipeline, debug=bool(job.debug_dir))
            out_img, result = res.image, res.result
            if job.debug_dir:
                dbg = _resolve(settings.storage_dir, job.debug_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                for name, img in res.debug.items():
                    cv2.imwrite(str(dbg / f"{name}.png"), img)

            out_path = _resolve(settings.storage_dir, job.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out_img)
            _resolve(settings.storage_dir, job.result_path).write_text(result.model_dump_json())
            status.update(status="done", bubbles=len(result.bubbles))
        except Exception as e:
            log.exception("job %s failed", job.job_id)
            status.update(status="failed", error=str(e))

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
