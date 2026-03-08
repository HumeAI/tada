import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_all_hellaswag(model_id="meta-llama/Llama-3.2-3B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use bfloat16 for speed and memory efficiency on modern GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map="auto"
    )
    model.eval()

    # 2. Load Dataset
    dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)

    correct = 0
    total = 0

    # 3. Processing Loop
    # We iterate through all 10,042 samples
    for example in tqdm(dataset, desc="Evaluating HellaSwag"):
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # Store losses for the 4 possible endings
        candidate_losses = []

        for ending in endings:
            # Construct the full sequence
            full_text = f"{ctx} {ending}"

            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Find where the context ends so we only calculate loss on the ending
            # We encode the context alone to find its length
            ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
            ctx_len = ctx_ids.shape[1]

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

                # Shift logits and labels for causal language modeling loss
                # Shifted Logits: [Batch, Seq_len - 1, Vocab]
                # Shifted Labels: [Batch, Seq_len - 1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                # Calculate cross-entropy loss per token
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # Only take the loss for the "ending" part of the sequence
                # We subtract 1 from ctx_len because of the shift
                ending_loss = loss[ctx_len - 1 :].mean()
                candidate_losses.append(ending_loss.item())

        # Prediction is the index with the minimum loss (highest likelihood)
        prediction = candidate_losses.index(min(candidate_losses))

        if prediction == label:
            correct += 1
        total += 1

        # 4. Final Results
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_all_hellaswag()
