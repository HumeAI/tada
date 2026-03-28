# MLX-TADA

TADA speech synthesis on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Also available on PyPI: `pip install mlx-tada`

## Setup

```bash
cd apple
uv venv
uv pip install -e .
```

For auto-transcription of reference audio (optional):
```bash
uv pip install mlx-whisper
```

## Weights

Download a reference audio clip:
```bash
curl -O "https://storage.googleapis.com/hume_reference_speakers/ljspeech.wav"
```

Pre-converted weights are downloaded and cached automatically. You still need [gated access to Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B) for the tokenizer:

```python
from mlx_tada import TadaForCausalLM, save_wav

model = TadaForCausalLM.from_pretrained("HumeAI/mlx-tada-3b", quantize=4)
ref = model.load_reference("ljspeech.wav")
out = model.generate("Hello, this is a test of TADA speech synthesis.", ref)
save_wav(out.audio, "output.wav")
```

Available models:
- [`HumeAI/mlx-tada-1b`](https://huggingface.co/HumeAI/mlx-tada-1b) — 1B English-only (~4.3 GB)
- [`HumeAI/mlx-tada-3b`](https://huggingface.co/HumeAI/mlx-tada-3b) — 3B multilingual (~8.9 GB)

### Offline Use

To download weights locally for offline inference:
```python
from huggingface_hub import snapshot_download
snapshot_download("HumeAI/mlx-tada-3b", local_dir="./weights/3b")
```

Then load from the local path:
```python
model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
```

## Generate Speech

### CLI

```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio ljspeech.wav \
  --text "The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think." \
  --output output.wav
```

With 4-bit quantization (10x faster, 60% less memory):
```bash
uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio ljspeech.wav \
  --text "The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think." \
  --quantize 4 \
  --output output.wav
```

### Python

```python
from mlx_tada import TadaForCausalLM, save_wav

model = TadaForCausalLM.from_pretrained("HumeAI/mlx-tada-3b", quantize=4)
ref = model.load_reference("ljspeech.wav")
out = model.generate("The history of artificial intelligence is a fascinating journey that spans decades of research and innovation. It all began in the 1950s when pioneers like Alan Turing first posed the question of whether machines could think.", ref)
save_wav(out.audio, "output.wav")

# out.audio     - numpy float32 array (24kHz)
# out.duration  - audio duration in seconds
# out.rtf       - real-time factor
# out.num_tokens
```

Supported language codes for `language`: `ar`, `ch`, `de`, `es`, `fr`, `it`, `ja`, `pl`, `pt`.
When omitted, MLX-TADA loads the default English aligner (`aligner`).

For multilingual alignment, pass `language` when loading weights (Python) or `--language` (CLI):
```python
model = TadaForCausalLM.from_pretrained("HumeAI/mlx-tada-3b", language="ja", quantize=4)
```

```bash
uv run python -m mlx_tada.generate --weights ./weights/3b --audio samples/ja_prompt.wav --language ja --audio-text "このムキムキのお兄さんがいるし バーだし少し高そうだと思いますよねこのバーの料金設定は良心的でした まあそんなに高くなかったです" --text "今日はとても良い天気ですね。散歩に行きましょう。"
```

### Inference Options

Control generation behavior with `InferenceOptions`:

```python
from mlx_tada import TadaForCausalLM, InferenceOptions, save_wav

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("ljspeech.wav")

opts = InferenceOptions(
    acoustic_cfg_scale=1.6,
    duration_cfg_scale=1.0,
    num_flow_matching_steps=10,
    time_schedule="logsnr",
    cfg_schedule="cosine",
)

out = model.generate(text="Hello world, today is a nice day.", reference=ref, inference_options=opts)
save_wav(out.audio, "output.wav")
```

The following inference options from the PyTorch version are **not currently supported** in MLX:
- `speed_up_factor`
- `num_acoustic_candidates`
- `scorer`
- `negative_condition_source`
- `text_only_logit_scale`
- `spkr_verification_weight`

### Speech Continuation

Use `num_extra_steps` to let the model generate speech beyond the provided text. The model continues speaking naturally and stops when it produces an end-of-sequence token:

```python
from mlx_tada import TadaForCausalLM, InferenceOptions, save_wav

model = TadaForCausalLM.from_weights("./weights/3b", quantize=4)
ref = model.load_reference("ljspeech.wav")

opts = InferenceOptions(
    acoustic_cfg_scale=1.6,
    num_flow_matching_steps=10,
    time_schedule="logsnr",
)

out = model.generate(
    text="The history of artificial intelligence is a fascinating journey that spans decades of research and innovation.",
    reference=ref,
    inference_options=opts,
    num_extra_steps=50,
)
save_wav(out.audio, "output.wav")
```

### Save and Reuse References

```python
from mlx_tada import Reference

ref = model.load_reference("ljspeech.wav")
ref.save("speaker.npz")

ref = Reference.load("speaker.npz")
out = model.generate("Reusing the same voice.", ref)
```

### Save Audio

```python
from mlx_tada import save_wav
save_wav(out.audio, "output.wav")
```

## Debug Logging

```bash
DEBUG=1 uv run python -m mlx_tada.generate \
  --weights ./weights/3b \
  --audio ljspeech.wav \
  --text "Hello"
```

```python
from mlx_tada import setup_logging

setup_logging()
```

## Running Tests

```bash
MLX_WEIGHTS=./weights/1b uv run pytest tests/ -v
```
