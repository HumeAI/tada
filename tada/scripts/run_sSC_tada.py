import os
import re
from collections import defaultdict

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tada.modules.encoder import Encoder, EncoderOutput
from tada.modules.tada import InferenceOptions, TadaForCausalLM


def load_samples(data_dir: str) -> dict[str, dict[str, str]]:
    """Discover sample IDs and map each to its correct/incorrect wav and txt paths."""
    files = os.listdir(data_dir)
    samples: dict[str, dict[str, str]] = defaultdict(dict)
    for fname in sorted(files):
        m = re.match(r"(.+)_(correct|incorrect)\.(wav|txt)$", fname)
        if m is None:
            continue
        sample_id, label, ext = m.group(1), m.group(2), m.group(3)
        key = f"{label}_{ext}"
        samples[sample_id.split("_", 1)[-1]][key] = os.path.join(data_dir, fname)
    return {k: v for k, v in samples.items() if "correct_wav" in v and "incorrect_wav" in v}


def compute_loss_audio(
    audio: torch.Tensor,
    model: TadaForCausalLM,
    encoder: Encoder,
    device: torch.device,
) -> float:
    """Encode audio, run the model, and return mean cross-entropy loss over all tokens."""
    audio = audio.to(device)
    prompt = encoder(
        audio=audio,
        text=None,
        sample_rate=24000,
        inference_window_size=30,
        inference_window_stride=28,
    )

    with torch.no_grad():
        outputs = model.generate(
            prompt,
            text="",
            num_transition_steps=0,
            num_extra_steps=0,
            inference_options=InferenceOptions(acoustic_cfg_scale=1.0, text_temperature=1.0, text_only_logit_scale=1.0),
            use_text_in_prompt=True,
            normalize_text=False,
        )

        shift_logits = outputs.logits[..., 7:-7, :].contiguous()
        shift_labels = outputs.input_text_ids[..., 8:-6].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return loss.mean().item()


def compute_loss_text_only(
    text: str,
    model: TadaForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Build a text-only EncoderOutput (zeroed acoustics) and return mean loss."""
    num_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    prompt = EncoderOutput(
        audio=torch.zeros(1, 0, device=device),
        audio_len=torch.zeros(1, device=device),
        text=[text],
        text_tokens_len=torch.tensor([num_tokens], device=device),
        token_positions=torch.zeros(1, num_tokens, dtype=torch.long, device=device),
        token_values=torch.zeros(1, num_tokens, model.config.acoustic_dim, device=device, dtype=model.dtype),
        token_masks=torch.zeros(1, num_tokens, dtype=torch.long, device=device),
    )

    with torch.no_grad():
        outputs = model.generate(
            prompt,
            text="",
            num_transition_steps=0,
            num_extra_steps=0,
            inference_options=InferenceOptions(acoustic_cfg_scale=1.0, text_only_logit_scale=100.0),
            use_text_in_prompt=True,
            normalize_text=False,
        )

        shift_logits = outputs.logits[..., 7:-7, :].contiguous()
        shift_labels = outputs.input_text_ids[..., 8:-6].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return loss.mean().item()


def _read_text(paths: dict[str, str], label: str, sample_id: str) -> str:
    txt_path = paths.get(f"{label}_txt")
    if txt_path is None:
        raise FileNotFoundError(
            f"Text file missing for {sample_id}_{label}; this mode requires .txt files alongside the .wav files"
        )
    with open(txt_path) as f:
        return f.read().strip()


def _load_wav(paths: dict[str, str], label: str) -> torch.Tensor:
    wav, sr = torchaudio.load(paths[f"{label}_wav"])
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    wav = wav.unsqueeze(0)  # (1, channels, samples)
    if wav.dim() == 3 and wav.shape[1] > 1:
        wav = wav.mean(dim=1, keepdim=True)
    return wav.squeeze(1)  # (1, samples)


def compute_loss_llama(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Standard causal LM loss over the full text sequence."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return loss.mean().item()


def evaluate_sSC(
    model_id: str = "HumeAI/tada-1b",
    data_dir: str = "/mnt/weka/trung/tSC/tSC",
    text_only: bool = False,
    llama: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "llama" if llama else ("text-only" if text_only else "audio")
    print(f"Using device: {device}")
    print(f"Mode: {mode}  Model: {model_id}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if llama:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        model = TadaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
    model.eval()

    encoder = None
    if not text_only and not llama:
        encoder = Encoder.from_pretrained("HumeAI/tada-codec")
        encoder = encoder.to(device)
        encoder.eval()

    samples = load_samples(data_dir)
    print(f"Found {len(samples)} samples in {data_dir}")

    correct = 0
    total = 0

    for sample_id in tqdm(sorted(samples), desc="Evaluating sSC"):
        paths = samples[sample_id]
        losses = {}

        for label in ("correct", "incorrect"):
            if llama:
                text = _read_text(paths, label, sample_id)
                losses[label] = compute_loss_llama(text, model, tokenizer, device)
            elif text_only:
                text = _read_text(paths, label, sample_id)
                losses[label] = compute_loss_text_only(text, model, tokenizer, device)
            else:
                wav = _load_wav(paths, label)
                losses[label] = compute_loss_audio(wav, model, encoder, device)

        if losses["correct"] < losses["incorrect"]:
            correct += 1
        total += 1

        accuracy = correct / total * 100
        print(
            f"[{total}] {sample_id}  correct_loss={losses['correct']:.4f}  "
            f"incorrect_loss={losses['incorrect']:.4f}  acc={accuracy:.2f}%"
        )

    print(f"\nFinal accuracy: {correct}/{total} = {correct / total * 100:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--data_dir", default="/mnt/weka/trung/tSC/tSC")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--llama", action="store_true", help="Use a plain Llama model with transcribed text")
    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = "meta-llama/Llama-3.2-1B" if args.llama else "HumeAI/tada-3b"

    evaluate_sSC(
        model_id=args.model_id,
        data_dir=args.data_dir,
        text_only=args.text_only,
        llama=args.llama,
    )
