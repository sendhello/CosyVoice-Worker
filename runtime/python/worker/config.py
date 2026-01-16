from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    """Configuration for the SQS worker service (Pydantic Settings v2).

    The settings are loaded from, in order of precedence:
    1) Explicit constructor kwargs
    2) Environment variables (optionally from a .env file)
    3) Default values below

    Environment variable names match prior implementation for backward
    compatibility, e.g. `SQS_QUEUE_URL`, `AWS_REGION`, etc.
    """

    # Pydantic Settings configuration
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # AWS SQS
    sqs_queue_url: str = Field(
        ..., validation_alias=AliasChoices('SQS_QUEUE_URL', 'sqs_queue_url')
    )
    aws_region: Optional[str] = Field(
        default=None, validation_alias=AliasChoices('AWS_REGION', 'aws_region')
    )
    aws_profile: Optional[str] = Field(
        default=None, validation_alias=AliasChoices('AWS_PROFILE', 'aws_profile')
    )

    # Polling / batching
    receive_max_messages: int = Field(
        10, validation_alias=AliasChoices('RECEIVE_MAX_MESSAGES', 'receive_max_messages')
    )
    wait_time_seconds: int = Field(
        20, validation_alias=AliasChoices('WAIT_TIME_SECONDS', 'wait_time_seconds')
    )
    visibility_timeout: int = Field(
        120, validation_alias=AliasChoices('VISIBILITY_TIMEOUT', 'visibility_timeout')
    )
    internal_queue_maxsize: int = Field(
        1000, validation_alias=AliasChoices('INTERNAL_QUEUE_MAXSIZE', 'internal_queue_maxsize')
    )
    gather_batch_max: int = Field(
        8, validation_alias=AliasChoices('GATHER_BATCH_MAX', 'gather_batch_max')
    )
    gather_batch_window_sec: float = Field(
        0.5, validation_alias=AliasChoices('GATHER_BATCH_WINDOW_SEC', 'gather_batch_window_sec')
    )
    vllm_batch_threshold: int = Field(
        4, validation_alias=AliasChoices('VLLM_BATCH_THRESHOLD', 'vllm_batch_threshold')
    )

    # CosyVoice model
    model_dir: str = Field(
        'pretrained_models/Fun-CosyVoice3-0.5B',
        validation_alias=AliasChoices('MODEL_DIR', 'model_dir'),
    )
    fp16: bool = Field(False, validation_alias=AliasChoices('FP16', 'fp16'))
    load_trt: bool = Field(False, validation_alias=AliasChoices('LOAD_TRT', 'load_trt'))
    trt_concurrent: int = Field(
        1, validation_alias=AliasChoices('TRT_CONCURRENT', 'trt_concurrent')
    )

    @staticmethod
    def from_env() -> "WorkerConfig":
        """Backward-compatible helper that loads settings from env/.env.

        Prefer direct `WorkerConfig()` construction.
        """
        return WorkerConfig()
