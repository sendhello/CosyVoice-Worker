from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from .base import Processor


class CosyVoiceVLLMProcessor(Processor):
    """Batch-oriented processor using CosyVoice model with vLLM backend.

    vLLM is expected to provide better throughput when many requests are
    processed in a short time window. This processor still supports processing
    a single payload, but shines when used with batches.
    """

    def __init__(self, model_dir: str, fp16: bool = False, load_trt: bool = False, trt_concurrent: int = 1) -> None:
        # load_vllm=True enables vLLM inside the CosyVoice model
        self.model = AutoModel(model_dir=model_dir, load_vllm=True, load_trt=load_trt, fp16=fp16, trt_concurrent=trt_concurrent)
        self.sample_rate = getattr(self.model, 'sample_rate', 24000)

    def _save_output(self, wav_tensor, output_path: str) -> None:
        torchaudio.save(output_path, wav_tensor, self.sample_rate)

    def process_one(self, payload: Dict[str, Any]) -> None:
        # Reuse the same logic as the single processor; vLLM will help internally
        mode = payload.get('mode', 'zero_shot')
        stream = bool(payload.get('stream', False))
        speed = float(payload.get('speed', 1.0))
        output_path = payload.get('output_path')

        if mode == 'sft':
            text = payload['tts_text']
            spk_id = payload['spk_id']
            for out in self.model.inference_sft(text, spk_id, stream=stream, speed=speed):
                if output_path:
                    self._save_output(out['tts_speech'], output_path)
        elif mode == 'zero_shot':
            text = payload['tts_text']
            prompt_text = payload.get('prompt_text', '')
            prompt_wav = payload['prompt_wav']
            for out in self.model.inference_zero_shot(text, prompt_text, prompt_wav, stream=stream, speed=speed):
                if output_path:
                    self._save_output(out['tts_speech'], output_path)
        elif mode == 'cross_lingual':
            text = payload['tts_text']
            prompt_wav = payload['prompt_wav']
            for out in self.model.inference_cross_lingual(text, prompt_wav, stream=stream, speed=speed):
                if output_path:
                    self._save_output(out['tts_speech'], output_path)
        elif mode == 'instruct':
            text = payload['tts_text']
            spk_id = payload.get('spk_id', '')
            instruct_text = payload['instruct_text']
            for out in self.model.inference_instruct(text, spk_id, instruct_text, stream=stream, speed=speed):
                if output_path:
                    self._save_output(out['tts_speech'], output_path)
        else:
            logging.warning(f"Unknown mode '{mode}', skipping message")

    def process_batch(self, payloads: Iterable[Dict[str, Any]]) -> None:
        # Note: The CosyVoice high-level API is iterator-based per item. We
        # still iterate, but vLLM engine inside the model can batch efficiently.
        for p in payloads:
            self.process_one(p)
