from __future__ import annotations

import threading
import time
from typing import List

from cosyvoice.utils.file_utils import logging

from .config import WorkerConfig
from .core.internal_queue import InternalQueue, WorkItem
from .messaging.sqs_client import SQSClient, SQSMessage
from .processing.base import Processor


class WorkerService:
    """SQS-driven worker service with internal batching.

    The service starts two threads:
    - consumer: polls SQS and pushes messages into an in-memory queue;
    - processor: gathers items in small time windows and decides whether to
      process them with a vLLM-backed processor (batch) or standard processor.
    """

    def __init__(self, config: WorkerConfig, single_processor: Processor, vllm_processor: Processor) -> None:
        self.cfg = config
        self.sqs = SQSClient(config.sqs_queue_url, config.aws_region, config.aws_profile)
        self.iq = InternalQueue(maxsize=config.internal_queue_maxsize)
        self.single = single_processor
        self.vllm = vllm_processor
        self._stop = threading.Event()
        self._consumer_t = threading.Thread(target=self._consumer_loop, name='sqs-consumer', daemon=True)
        self._processor_t = threading.Thread(target=self._processor_loop, name='processor', daemon=True)

    def start(self) -> None:
        logging.info('Starting worker service...')
        self._consumer_t.start()
        self._processor_t.start()

    def stop(self, timeout: float = 10.0) -> None:
        logging.info('Stopping worker service...')
        self._stop.set()
        self._consumer_t.join(timeout=timeout)
        self._processor_t.join(timeout=timeout)

    # --- Internal loops ---
    def _consumer_loop(self) -> None:
        logging.info('SQS consumer loop started')
        while not self._stop.is_set():
            try:
                msgs: List[SQSMessage] = self.sqs.receive(
                    max_messages=self.cfg.receive_max_messages,
                    wait_time_seconds=self.cfg.wait_time_seconds,
                    visibility_timeout=self.cfg.visibility_timeout,
                )
                if not msgs:
                    continue
                for m in msgs:
                    item = WorkItem(payload=m.body, receipt_handle=m.receipt_handle, message_id=m.message_id)
                    try:
                        self.iq.put(item)
                    except Exception:
                        # If internal queue is full, wait a bit and retry once
                        time.sleep(0.1)
                        try:
                            self.iq.put(item, timeout=1.0)
                        except Exception:
                            logging.warning('Internal queue full, dropping message %s', m.message_id)
            except Exception as e:
                logging.error('SQS consumer loop error: %s', e)
                time.sleep(1.0)

    def _processor_loop(self) -> None:
        logging.info('Processor loop started')
        while not self._stop.is_set():
            try:
                batch = self.iq.gather(self.cfg.gather_batch_max, self.cfg.gather_batch_window_sec)
                if not batch:
                    continue
                receipt_handles = [w.receipt_handle for w in batch]
                payloads = [w.payload for w in batch]
                if len(batch) >= self.cfg.vllm_batch_threshold:
                    logging.info('Processing batch of %d with vLLM', len(batch))
                    self.vllm.process_batch(payloads)
                else:
                    logging.info('Processing %d item(s) with single processor', len(batch))
                    if len(batch) == 1:
                        self.single.process_one(payloads[0])
                    else:
                        # There is no explicit batch API for the standard model; just loop
                        for p in payloads:
                            self.single.process_one(p)
                # Ack on success
                if len(receipt_handles) == 1:
                    self.sqs.delete(receipt_handles[0])
                else:
                    self.sqs.delete_batch(receipt_handles)
            except Exception as e:
                # On failure, do not ack; message will reappear after visibility timeout
                logging.error('Processor loop error: %s', e)
                time.sleep(0.1)
