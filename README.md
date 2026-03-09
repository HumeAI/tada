<!-- ---
license: mit
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
  <a href="https://huggingface.co/collections/HumeAI/tada-684390a52ed08a7717b5e702"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow" alt="Collection"></a>
  <a href="https://pypi.org/project/hume-tada/"><img src="https://img.shields.io/badge/PyPI-hume--tada-3775A9.svg?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/HumeAI/tada/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<img width="2400" height="1260" alt="image" src="https://github.com/user-attachments/assets/800eb8c5-eb6f-4e03-b8f3-150055a6cdfc" />

<p align="center"><br/><em>A unified speech-language model that synchronizes speech and text into a single, cohesive stream via 1:1 alignment.</em></p>

---

TADA achieves high-fidelity synthesis and generation with a fraction of the computational overhead required by traditional models. By leveraging a novel tokenizer and architectural design, each autoregressive step covers one text token, dynamically determining its duration and prosody — eliminating fixed frame rates and transcript hallucination.

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
    <td><img src="figures/MOS.png" alt="MOS" height="300px"></td>
  </tr>
  <tr>
    <td><img src="figures/real-time.png" alt="Speed" height="300px"></td>
    <td><img src="figures/speaker-sim.png" alt="Speaker Similarity" height="300px"></td>
  </tr>
</table>

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

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

device = "cuda"
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
model = TadaForCausalLM.from_pretrained("HumeAI/tada-3b-ml").to(device)

audio, sample_rate = torchaudio.load("samples/ljspeech.wav")
audio = audio.to(device)
prompt_text = "The examination and testimony of the experts, enabled the commission to conclude that five shots may have been fired."
prompt = encoder(
    audio, text=[prompt_text], sample_rate=sample_rate
)

output = model.generate(
    prompt=prompt,
    text="Please call Stella. Ask her to bring these things with her from the store.",
)
```

### Speech continuation

Provide `num_extra_steps` if you want to generate text-speech continuation of the prompt:

```python
output = model.generate(
    prompt=prompt,
    num_extra_steps=50
)
```

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
