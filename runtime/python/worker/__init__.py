"""Worker package for SQS-driven CosyVoice processing.

This package contains building blocks to run a background worker that:
- pulls tasks from an AWS SQS queue,
- buffers them in an internal in-memory queue,
- processes tasks in batches with a vLLM-backed CosyVoice model when there are enough tasks,
- otherwise processes tasks one-by-one with the standard CosyVoice model.

All public classes and functions are documented in English.
"""
