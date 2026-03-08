import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tada.modules.encoder import EncoderOutput
from tada.modules.tada import InferenceOptions, TadaForCausalLM


def evaluate_storycloze(model_id="HumeAI/tada-3b"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    model = TadaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map="auto"
    )
    model.eval()

    dataset = load_dataset("LSDSem/story_cloze", "2016", split="validation", trust_remote_code=True)

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating StoryCloze"):
        ctx = " ".join(
            [
                example["input_sentence_1"],
                example["input_sentence_2"],
                example["input_sentence_3"],
                example["input_sentence_4"],
            ]
        )
        endings = [example["sentence_quiz1"], example["sentence_quiz2"]]
        label = int(example["answer_right_ending"]) - 1  # 1-indexed → 0-indexed

        ctx_len = len(tokenizer.encode(ctx, add_special_tokens=False))
        candidate_losses = []

        for ending in endings:
            full_text = f"{ctx} {ending}"
            num_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

            prompt = EncoderOutput(
                audio=torch.zeros(1, 0, device=device),
                audio_len=torch.zeros(1, device=device),
                text=[full_text],
                text_tokens_len=torch.tensor([num_tokens], device=device),
                token_positions=torch.zeros(1, num_tokens, dtype=torch.long, device=device),
                token_values=torch.zeros(1, num_tokens, model.config.acoustic_dim, device=device, dtype=model.dtype),
            )

            with torch.no_grad():
                outputs = model.generate(
                    prompt,
                    text="",
                    num_transition_steps=0,
                    num_extra_steps=0,
                    inference_options=InferenceOptions(acoustic_cfg_scale=1.0),
                    use_text_in_prompt=True,
                    normalize_text=False,
                )

                shift_logits = outputs.logits[..., 7:-7, :].contiguous()
                shift_labels = outputs.input_text_ids[..., 8:-6].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                ending_loss = loss[ctx_len:].mean()
                candidate_losses.append(ending_loss.item())

        prediction = candidate_losses.index(min(candidate_losses))

        if prediction == label:
            correct += 1
        total += 1

        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_storycloze()
