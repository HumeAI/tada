<!-- ---
license: llama3.2
language:
  - en
tags:
  - tts
  - text-to-speech
  - speech-language-model
--- -->

<h1 align="center">TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.23068"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/spaces/HumeAI/tada"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Demo"></a>
  <a href="https://www.hume.ai/blog/opensource-tada"><img src="https://img.shields.io/badge/Blog-Post-orange.svg" alt="Blog"></a>
  <a href="https://huggingface.co/collections/HumeAI/tada"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow" alt="Collection"></a>
  <a href="https://pypi.org/project/hume-tada/"><img src="https://img.shields.io/badge/PyPI-hume--tada-3775A9.svg?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/HumeAI/tada/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<img width="2400" height="1260" alt="image" src="https://github.com/user-attachments/assets/800eb8c5-eb6f-4e03-b8f3-150055a6cdfc" />

<p align="center"><br/><em>A unified speech-language model that synchronizes speech and text into a single, cohesive stream via 1:1 alignment.</em></p>

---

TADA achieves high-fidelity synthesis and generation with a fraction of the computational overhead required by traditional models. By leveraging a novel tokenizer and architectural design, each autoregressive step covers one text token, dynamically determining its duration and prosody — eliminating fixed frame rates and transcript hallucination.

## Updates

**March 2026**
- **Streaming audio generation** — `generate(stream=True)` for end-to-end streaming. Audio chunks are yielded during LLM generation. True TTFA ~350ms on GPU. No retraining needed.
- Encoder no longer loaded inside `TadaForCausalLM` — saves ~2.5 GB VRAM. Load it separately only when encoding new prompts.
- Added `EncoderOutput.save()` / `EncoderOutput.load()` for prompt caching — encode once, reuse without the encoder.
- Default flow matching steps reduced from 20 to 10 (no perceptible quality loss, ~1.3x faster).
- bf16 inference support via `torch_dtype=torch.bfloat16` — halves model memory (~9 GB for 3B).
- `model.compile()` for torch.compile optimization — ~0.12x RTF on H100 with cached prompts.

## Key Features

- **1:1 Token Alignment** — The tokenizer encodes audio into a sequence of vectors that perfectly matches the number of text tokens.
- **Dynamic Duration Synthesis** — Generates the full speech segment for a text token in a single autoregressive step, regardless of length.
- **Dual-Stream Generation** — Generates a text token and the speech for the preceding token simultaneously, maintaining the same context length as text-only generation.
- **Efficiency & Reliability** — Superior expressiveness and natural flow while significantly reducing computational cost.

## How It Works

### The Tokenization Schema

TADA unifies modalities by ensuring that for every word or subword token, there is exactly one corresponding speech vector. This synchronized stream allows the model to "understand" the precise timing of speech relative to text.

### Dynamic Autoregression

Most TTS models require a fixed number of steps to produce one second of audio (e.g., 50 frames per second). TADA breaks this constraint:

- Each autoregressive step covers one text token.
- The model dynamically determines the duration and prosody for that specific token.
- This results in a more natural flow and eliminates transcript hallucination.

## Evaluation

<table>
  <tr>
    <td><img src="figures/CER.png" alt="CER" height="300px"></td>
    <td><img src="figures/real-time.png" alt="Speed" height="300px"></td>
  </tr>
  <tr>
    <td><img src="figures/naturalness.png" alt="Naturalness MOS" height="280px"></td>
    <td><img src="figures/speaker-sim.png" alt="Speaker Similarity" height="270px"></td>
  </tr>
</table>

## Prerequisites

TADA models are built on [Meta Llama 3.2](https://huggingface.co/meta-llama). You must request access to the Llama models before using TADA:

- Visit [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) or [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and accept the license agreement

## Installation

```bash
pip install hume-tada
```

### Build from source

```bash
git clone https://github.com/HumeAI/tada.git
cd tada
pip install -e .
```

## Models

| Model | Base Model | HuggingFace Hub |
|-------|-----------|-----------------|
| TADA-1B | Llama 3.2 1B | [`HumeAI/tada-1b`](https://huggingface.co/HumeAI/tada-1b) |
| TADA-3B-ML | Llama 3.2 3B | [`HumeAI/tada-3b-ml`](https://huggingface.co/HumeAI/tada-3b-ml) |

All models use the same encoder ([`HumeAI/tada-codec`](https://huggingface.co/HumeAI/tada-codec)) and can be loaded using the same API.

## Run Inference

### Text-to-Speech

```python
import torch
import torchaudio

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import TadaForCausalLM

device = "cuda"

# Encoder is loaded separately (not inside the model)
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
model = TadaForCausalLM.from_pretrained("HumeAI/tada-3b-ml", torch_dtype=torch.bfloat16).to(device)

audio, sample_rate = torchaudio.load("samples/ljspeech.wav")
audio = audio.to(device)
prompt_text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
prompt = encoder(
    audio, text=[prompt_text], sample_rate=sample_rate
)

# Optional: save prompt to skip encoder on future runs
# prompt.save("prompt_cache.pt")
# prompt = EncoderOutput.load("prompt_cache.pt", device=device)

output = model.generate(
    prompt=prompt,
    text="Please call Stella. Ask her to bring these things with her from the store.",
)
```

### Multilingual Generation

TADA supports multilingual speech synthesis via language-specific aligners. Pass the `language` parameter when loading the encoder to use the appropriate aligner for your target language.

```python
import torch
import torchaudio

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

device = "cuda"
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder", language="ja").to(device)
model = TadaForCausalLM.from_pretrained("HumeAI/tada-3b-ml", torch_dtype=torch.bfloat16).to(device)

# Load a reference audio clip in the target language
audio, sample_rate = torchaudio.load("samples/ja_prompt.wav")
audio = audio.to(device)

# For non-English prompts, provide the transcript so the encoder uses forced alignment
# instead of the built-in ASR (which is English-only)
prompt_text = "このムキムキのお兄さんがいるし バーだし少し高そうだと思いますよねこのバーの料金設定は良心的でした まあそんなに高くなかったです"
prompt = encoder(audio, text=[prompt_text], sample_rate=sample_rate)

output = model.generate(
    prompt=prompt,
    text="今日はとても良い天気ですね。散歩に行きましょう。",
)
```

Supported languages: `ar`, `ch`, `de`, `es`, `fr`, `it`, `ja`, `pl`, `pt`. When `language` is not specified, the default English aligner is used.

> **Note:** For non-English prompts, you should provide the transcript of the reference audio via the `text` parameter. The encoder's built-in ASR is English-only. The generation will still work, but alignment quality will be degraded.

You can inspect the prompt alignment to verify it looks correct:

```python
prompt.print_alignment(model.tokenizer)
```

This shows a dot-span visualization of the token-to-audio alignment — dots represent frame gaps, tokens appear at their aligned positions:

```
34 tokens | 10.50s audio
······The··exam····ination··and·····test···imony··of···the
```

- If alignment looks wrong (tokens bunched together, missing tokens, nonsensical text), check that you provided the correct transcript.
- This is especially important for non-English prompts where the built-in ASR cannot be used.

### Speech continuation

Provide `num_extra_steps` if you want to generate text-speech continuation of the prompt:

```python
output = model.generate(
    prompt=prompt,
    num_extra_steps=50
)
```


### Streaming Audio Generation

By default, `generate()` produces the complete audio after all tokens are generated (non-streaming). To stream audio chunks as they are generated, pass `stream=True`:

```python
import torch
import torchaudio

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import TadaForCausalLM

device = "cuda"
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b", torch_dtype=torch.bfloat16).to(device)

audio, sample_rate = torchaudio.load("samples/ljspeech.wav")
prompt = encoder(audio.to(device), sample_rate=sample_rate)

# Stream audio chunks as they are generated
stream = model.generate(
    prompt=prompt,
    text="Hello, this is a demonstration of streaming text to speech.",
    stream=True,
)
for chunk, sample_rate in stream:
    # chunk shape: (num_samples,) at 24kHz
    # Write to audio buffer, stream to speaker, send over network, etc.
    print(f"Received {chunk.shape[-1] / sample_rate:.2f}s of audio")

# After iteration, stream.result contains the full GenerationOutput
# (audio field is None since chunks were streamed individually)
output = stream.result
```

When `stream` is not set (default), the existing non-streaming path runs unchanged.

#### Application integration

For production applications, connect the iterator to your audio pipeline:

```python
import queue

# Thread-safe audio queue for a web server
audio_queue = queue.Queue()

stream = model.generate(prompt=prompt, text="Hello world", stream=True)
for chunk, sr in stream:
    audio_bytes = (chunk.cpu().numpy() * 32767).astype("int16").tobytes()
    audio_queue.put(audio_bytes)
audio_queue.put(None)  # Signal end of stream
```

```python
# Write chunks directly to a WAV file as they arrive
import soundfile as sf

with sf.SoundFile("output.wav", mode="w", samplerate=24000, channels=1) as f:
    for chunk, sr in model.generate(prompt=prompt, text="Hello world", stream=True):
        f.write(chunk.cpu().numpy())
```

For CPU or memory-constrained devices, reduce the CNN window size:

```python
stream = model.generate(
    prompt=prompt,
    text="Hello world",
    stream=True,
    streaming_cnn_window_size=50,  # Default 100. Use 50 for <32GB RAM.
)
```

#### How it works

Streaming is end-to-end: audio chunks are emitted **during** LLM generation, not after. As the LLM generates each token, the acoustic features are immediately decoded into audio and yielded to the caller.

The decoder has two stages:

1. **Transformer**: Per-layer KV cache ensures each new token only computes Q and attends to cached K,V. This is bit-exact with the non-streaming path.

2. **CNN (DACDecoder)**: A sliding window (default 100 frames) over accumulated transformer hidden states. The window provides 20 frames of left context and 15 frames of right lookahead for stable output. No model retraining required.

#### Performance

True time-to-first-audio (TTFA) measured from `generate()` call to first audio chunk:

| Model | Device | Text | True TTFA | Total time | Non-streaming |
|-------|--------|------|-----------|------------|---------------|
| TADA-1B | GPU (H100) | Short | ~440ms | 0.47s | 0.77s |
| TADA-1B | GPU (H100) | Medium | ~350ms | 1.06s | 0.86s |
| TADA-1B | GPU (H100) | Long | ~330ms | 3.02s | 2.47s |
| TADA-1B | CPU | Short | ~2.3s | 3.2s | 2.9s |
| TADA-1B | CPU | Long | ~2.1s | 25s | 18s |

Streaming uses less peak memory than non-streaming for medium and long text (up to 3.6 GB less VRAM on GPU).

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stream` | `False` | When `True`, returns an `AudioStream` iterator yielding `(chunk, sample_rate)` tuples. |
| `streaming_cnn_window_size` | 100 | CNN sliding window size in frames. Use 50 for <32GB RAM. |

## License

This repository contains both model weights and code, which are licensed separately:

- **Model weights** are licensed under the Llama 3.2 Community License Agreement  
- **Code** in this repository is licensed under the MIT License  

You must comply with the terms of the Llama 3.2 license when using the models.

See:
- `LICENSE` for the Llama 3.2 license
- `LICENSE_CODE` for the MIT license

## Acknowledgements

This project is built using Llama 3.2.

Llama 3.2 is licensed under the Llama 3.2 Community License.

## 📚 Citation

If you use this project in your research, please cite our paper:

```bibtex
@article{dang2026tada,
  title={TADA: A Generative Framework for Speech Modeling via Text-Acoustic Dual Alignment},
  author={Dang, Trung and Rao, Sharath and Gupta, Ananya and Gagne, Christopher and Tzirakis, Panagiotis and Baird, Alice and Cłapa, Jakub Piotr and Chin, Peter and Cowen, Alan},
  journal={arXiv preprint arXiv:2602.23068},
  year={2026}
}
```

## Contact

[Hume AI](https://hume.ai) is an empathic AI research company. We research the datasets, tools, and models needed to give empathy to AI models to serve human wellbeing. If you're interested in any of our product or research collaborations, please reach out to us at hello@hume.ai.
