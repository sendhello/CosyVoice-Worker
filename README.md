
We strongly recommend that you download our pretrained `Fun-CosyVoice3-0.5B` `CosyVoice2-0.5B` `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource.

``` python
# modelscope SDK model download
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# for oversea users, huggingface SDK model download
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('FunAudioLLM/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use wetext by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Basic Usage

We strongly recommend using `Fun-CosyVoice3-0.5B` for better performance.
Follow the code in `example.py` for detailed usage of each model.
```sh
python example.py
```

#### vLLM Usage
CosyVoice2/3 now supports **vLLM 0.11.x+ (V1 engine)** and **vLLM 0.9.0 (legacy)**.
Older vllm version(<0.9.0) do not support CosyVoice inference, and versions in between (e.g., 0.10.x) are not tested.

Notice that `vllm` has a lot of specific requirements. You can create a new env to in case your hardward do not support vllm and old env is corrupted.

``` sh
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
# for vllm==0.9.0
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# for vllm>=0.11.0
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py
```

#### Start web demo

You can use our web demo page to get familiar with CosyVoice quickly.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

#### Advanced Usage

For advanced users, we have provided training and inference scripts in `examples/libritts`.

#### Build for deployment

Optionally, if you want service deployment,
You can run the following steps.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

#### Using Nvidia TensorRT-LLM for deployment

Using TensorRT-LLM to accelerate cosyvoice2 llm could give 4x acceleration comparing with huggingface transformers implementation.
To quick start:

``` sh
cd runtime/triton_trtllm
docker compose up -d
```
For more details, you could check [here](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Citations

``` bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.


## SQS Worker (Background Processing)

The repository includes an optional SQS-driven worker that pulls tasks from an AWS SQS queue, buffers them in an internal in-memory queue, and processes them either:
- in batches with a vLLM-enabled CosyVoice model (higher throughput), or
- one-by-one with the standard CosyVoice model (lower latency) when few tasks are available.

### Installation

Make sure the base dependencies are installed (see Install section). Additionally, the worker requires:

```
pip install boto3
# vLLM is optional but recommended for batching; install if you plan to enable it in the model
# pip install vllm
```

### Message format (SQS Body)

Messages are JSON objects. The minimal fields depend on the `mode`:

```
{
  "mode": "zero_shot",              // one of: sft | zero_shot | cross_lingual | instruct
  "tts_text": "Hello world",        // text to synthesize
  "prompt_text": "",                // optional, used by zero_shot
  "prompt_wav": "/path/to.wav",     // local path or URL (e.g., s3://bucket/key.wav)
  "spk_id": "",                     // used by sft/instruct (optional)
  "instruct_text": "",              // used by instruct (optional)
  "speed": 1.0,                      // optional, default 1.0
  "stream": false,                   // optional, default false
  "output_path": "/tmp/out.wav"     // where to save the resulting WAV
}
```

Notes:
- The worker will save the generated audio to `output_path` (WAV). Ensure the path is writable.
- If you use `s3://...` URIs for inputs/outputs, you may extend the worker to download/upload; by default it expects local paths.

### Running the worker

You can configure via environment variables or CLI flags.

Required:
- `SQS_QUEUE_URL` – your SQS queue URL

Optional:
- `AWS_REGION`, `AWS_PROFILE` – AWS config
- `MODEL_DIR` – path to the model directory (defaults to `pretrained_models/Fun-CosyVoice3-0.5B`)
- `FP16`, `LOAD_TRT`, `TRT_CONCURRENT` – model performance tuning
- Batching knobs: `RECEIVE_MAX_MESSAGES`, `WAIT_TIME_SECONDS`, `VISIBILITY_TIMEOUT`, `INTERNAL_QUEUE_MAXSIZE`, `GATHER_BATCH_MAX`, `GATHER_BATCH_WINDOW_SEC`, `VLLM_BATCH_THRESHOLD`

Example (env):

```
export SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/tts-tasks
export AWS_REGION=us-east-1
export MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B
python sqs_worker.py
```

Example (CLI):

```
python sqs_worker.py \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/tts-tasks \
  --aws-region us-east-1 \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
  --vllm-batch-threshold 4 \
  --gather-batch-max 8 \
  --gather-batch-window-sec 0.5
```

### Behavior

- The consumer thread long-polls SQS (up to 20s) and pushes tasks into an internal queue.
- The processor thread gathers small batches within a short time window (`GATHER_BATCH_WINDOW_SEC`).
- If the gathered batch size is `>= VLLM_BATCH_THRESHOLD`, the worker uses the vLLM-enabled model; otherwise it processes items one-by-one using the standard model.
- Messages are acknowledged (deleted) from SQS only after successful processing. On errors, messages are not deleted and will reappear after the visibility timeout, following SQS redrive policy if configured.
