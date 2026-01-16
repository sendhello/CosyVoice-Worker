#!/usr/bin/env python3
"""SQS-driven worker for CosyVoice.

This worker pulls tasks from AWS SQS, buffers them in an internal queue,
and processes them either in batches using a vLLM-backed model, or one-by-one
using the standard model if the batch is too small.

Configuration is read from environment variables by default; see README for
details. You can also override via CLI flags.
"""

import argparse
import os
import signal
import sys
from contextlib import contextmanager

from runtime.python.worker.config import WorkerConfig
from runtime.python.worker.processing.cosyvoice_single import CosyVoiceSingleProcessor
from runtime.python.worker.processing.cosyvoice_vllm import CosyVoiceVLLMProcessor
from runtime.python.worker.service import WorkerService
from cosyvoice.utils.file_utils import logging


def _build_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    """Merge env/.env with CLI overrides using Pydantic Settings.

    Values passed via CLI take precedence over environment variables; anything
    not provided falls back to env/.env, then defaults.
    """
    overrides = {}
    if args.queue_url is not None:
        overrides['sqs_queue_url'] = args.queue_url
    if args.aws_region is not None:
        overrides['aws_region'] = args.aws_region
    if args.aws_profile is not None:
        overrides['aws_profile'] = args.aws_profile
    if args.model_dir is not None:
        overrides['model_dir'] = args.model_dir
    if args.fp16 is not None:
        overrides['fp16'] = args.fp16
    if args.load_trt is not None:
        overrides['load_trt'] = args.load_trt
    if args.trt_concurrent is not None:
        overrides['trt_concurrent'] = args.trt_concurrent
    if args.receive_max_messages is not None:
        overrides['receive_max_messages'] = args.receive_max_messages
    if args.wait_time_seconds is not None:
        overrides['wait_time_seconds'] = args.wait_time_seconds
    if args.visibility_timeout is not None:
        overrides['visibility_timeout'] = args.visibility_timeout
    if args.internal_queue_maxsize is not None:
        overrides['internal_queue_maxsize'] = args.internal_queue_maxsize
    if args.gather_batch_max is not None:
        overrides['gather_batch_max'] = args.gather_batch_max
    if args.gather_batch_window_sec is not None:
        overrides['gather_batch_window_sec'] = args.gather_batch_window_sec
    if args.vllm_batch_threshold is not None:
        overrides['vllm_batch_threshold'] = args.vllm_batch_threshold

    try:
        cfg = WorkerConfig(**overrides)
    except Exception as e:
        # The most likely validation error is a missing SQS queue URL
        raise SystemExit(f'Failed to build config: {e}. Provide --queue-url or set SQS_QUEUE_URL')
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CosyVoice SQS Worker')
    p.add_argument('--queue-url', type=str, default=None, help='SQS queue URL (or set SQS_QUEUE_URL)')
    p.add_argument('--aws-region', type=str, default=None, help='AWS region (or set AWS_REGION)')
    p.add_argument('--aws-profile', type=str, default=None, help='AWS profile (or set AWS_PROFILE)')
    p.add_argument('--model-dir', type=str, default=None, help='Model directory (or set MODEL_DIR)')
    p.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=None, help='Enable fp16')
    p.add_argument('--load-trt', action=argparse.BooleanOptionalAction, default=None, help='Enable TensorRT')
    p.add_argument('--trt-concurrent', type=int, default=None, help='TensorRT concurrent streams')
    p.add_argument('--receive-max-messages', type=int, default=None)
    p.add_argument('--wait-time-seconds', type=int, default=None)
    p.add_argument('--visibility-timeout', type=int, default=None)
    p.add_argument('--internal-queue-maxsize', type=int, default=None)
    p.add_argument('--gather-batch-max', type=int, default=None)
    p.add_argument('--gather-batch-window-sec', type=float, default=None)
    p.add_argument('--vllm-batch-threshold', type=int, default=None)
    return p.parse_args(argv)


@contextmanager
def _graceful_shutdown(service: WorkerService):
    def handler(signum, frame):
        logging.info('Signal %s received, shutting down...', signum)
        service.stop()
        sys.exit(0)

    old_int = signal.signal(signal.SIGINT, handler)
    old_term = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _build_config_from_args(args)
    logging.info('Starting worker with config: %s', cfg)

    single = CosyVoiceSingleProcessor(cfg.model_dir, fp16=cfg.fp16, load_trt=cfg.load_trt, trt_concurrent=cfg.trt_concurrent)
    vllm = CosyVoiceVLLMProcessor(cfg.model_dir, fp16=cfg.fp16, load_trt=cfg.load_trt, trt_concurrent=cfg.trt_concurrent)
    service = WorkerService(cfg, single_processor=single, vllm_processor=vllm)

    with _graceful_shutdown(service):
        service.start()
        # Block forever; threads do the work
        while True:
            signal.pause()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
