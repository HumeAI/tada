from dataclasses import dataclass

import torch
import torchaudio
from transformers import AutoConfig, AutoModelForCTC, AutoTokenizer, PretrainedConfig, PreTrainedModel


def _align_text_tokens(probs: torch.Tensor, text_tokens: torch.Tensor) -> list[int]:
    """Align text tokens to the probs.

    Args:
        probs: shape L x V where L is the length of the time aligned sequence and V is the vocabulary size.
        text_tokens: tensor of T text tokens.

    Returns:
        The assigned positions such that the sum of the probs of the assigned positions is maximized.
    """

    # F[i][j]: best score from aligning text_tokens[:j + 1] with probs[:i + 1]
    L, V = probs.shape
    T = len(text_tokens)
    device = probs.device

    F = torch.full((L, T), -float("inf"), device=device)
    backpointer = torch.zeros(L, T, dtype=torch.long, device=device)

    # Extract probabilities for all tokens at once
    token_probs = probs[:, text_tokens]  # Shape: (L, T)

    # Initialize first column: find best position for first token (vectorized)
    cummax_first = torch.cummax(token_probs[:, 0], dim=0)
    F[:, 0] = cummax_first.values
    backpointer[:, 0] = cummax_first.indices

    # Initialize diagonal: forced alignment where each token gets consecutive positions
    if T <= L:
        diag_indices = torch.arange(T, device=device)
        F[diag_indices, diag_indices] = torch.cumsum(token_probs[diag_indices, diag_indices], dim=0)
        backpointer[diag_indices, diag_indices] = diag_indices

    # Fill DP table (still needs a loop due to dependencies, but optimized)
    for i in range(1, L):
        max_j = min(i, T)
        if max_j <= 1:
            continue

        # Vectorize the inner loop for all j at position i
        j_range = torch.arange(1, max_j, device=device)

        # Choice 1: skip position i
        skip_scores = F[i - 1, j_range]

        # Choice 2: use position i for token j
        use_scores = F[i - 1, j_range - 1] + token_probs[i, j_range]

        # Compare and update
        use_better = use_scores >= skip_scores
        F[i, j_range] = torch.where(use_better, use_scores, skip_scores)
        backpointer[i, j_range] = torch.where(use_better, i, -1)

    # Traceback to find positions (keep as loop but minimize operations)
    positions = torch.zeros(T, dtype=torch.long, device=device)
    i, j = L - 1, T - 1
    pos_idx = T - 1

    while j >= 0:
        if j == 0:
            positions[pos_idx] = backpointer[i, j]
            break
        elif backpointer[i, j] == -1:
            i -= 1
        else:
            positions[pos_idx] = backpointer[i, j]
            pos_idx -= 1
            i -= 1
            j -= 1

    return positions.tolist()


class AlignerConfig(PretrainedConfig):
    base_model_name: str = "facebook/wav2vec2-large"
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"
    emb_dim: int = 4096


@dataclass
class AlignOutput:
    token_positions: torch.Tensor
    token_masks: torch.Tensor
    logits: torch.Tensor | None = None


class Aligner(PreTrainedModel):
    config_class = AlignerConfig

    def __init__(
        self,
        config: AlignerConfig,
    ):
        """Aligner module for aligning audio to text tokens.

        Args:
            base_model_name: Name of the base model to use for the encoder.
            tokenizer_name: Name of the tokenizer to use for the tokenizer.
            secondary_tokenizer_name: Name of the secondary tokenizer to use for the secondary tokenizer.
            emb_dim: Dimension of the embedding.
            inference_window_size: Window size for sliding window inference (in audio samples at 16kHz after resampling).
            inference_window_stride: Stride between windows (default: window_size // 2).
        """
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        self.encoder_config = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder_config.vocab_size = len(self.tokenizer)
        self.encoder = AutoModelForCTC.from_config(self.encoder_config)

    def _forward_encoder_window(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """Forward pass for encoder on a single window."""
        return self.encoder(audio, attention_mask=attention_mask).logits

    def _forward_encoder_sliding_window(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inference_window_size: float | None = None,
        inference_window_stride: float | None = None,
    ):
        """Forward pass using sliding window inference for the encoder.

        Args:
            audio: (batch_size, audio_len) - audio at 16kHz
            attention_mask: (batch_size, audio_len)
            inference_window_size: Size of the sliding window for long audio.
            inference_window_stride: Stride of the sliding window for long audio.
        Returns:
            Output with logits that have been averaged over overlapping windows
        """
        batch_size, audio_len = audio.shape
        window_size = int(inference_window_size * 16000)
        stride = int(inference_window_stride * 16000)
        downsample_ratio = 320
        vocab_size = self.encoder_config.vocab_size
        # If sequence fits in one window, use regular forward
        if audio_len <= window_size:
            return self._forward_encoder_window(audio, attention_mask)

        # Estimate output length
        output_len = int(audio_len / downsample_ratio)

        # Prepare output tensors
        logits_sum = torch.zeros(batch_size, output_len, vocab_size, device=audio.device, dtype=audio.dtype)
        counts = torch.zeros(batch_size, output_len, 1, device=audio.device, dtype=audio.dtype)

        # Slide through the sequence
        for start_idx in range(0, audio_len, stride):
            end_idx = min(start_idx + window_size, audio_len)

            # Extract window
            audio_window = audio[:, start_idx:end_idx]

            # Extract attention mask window if provided
            mask_window = None
            if attention_mask is not None:
                mask_window = attention_mask[:, start_idx:end_idx]

            # Process window
            logits_window = self._forward_encoder_window(audio_window, mask_window)

            # Calculate output positions for this window
            out_start = int(start_idx / downsample_ratio)
            out_end = out_start + logits_window.shape[1]
            out_end = min(out_end, output_len + 1)
            actual_out_len = out_end - out_start

            # Accumulate results (handle potential size mismatch)
            logits_sum[:, out_start:out_end, :] += logits_window[:, :actual_out_len, :]
            counts[:, out_start:out_end, :] += 1

            # If we've reached the end, break
            if end_idx >= audio_len:
                break

        # Average overlapping regions
        logits = logits_sum / counts.clamp(min=1)

        return logits

    def forward(
        self,
        audio: torch.Tensor,
        text: list[str] | None = None,
        text_tokens: torch.Tensor | None = None,
        audio_length: torch.Tensor | None = None,
        sample_rate: int = 24000,
        return_logits: bool = False,
        inference_window_size: int | None = None,
        inference_window_stride: int | None = None,
    ) -> AlignOutput:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        attention_mask = None
        if audio_length is not None:
            attention_mask = torch.arange(audio.shape[1], device=audio.device).unsqueeze(0) < audio_length

        # Use sliding window inference if configured and not in training mode
        if inference_window_size is not None and not self.training:
            logits = self._forward_encoder_sliding_window(
                audio,
                attention_mask=attention_mask,
                inference_window_size=inference_window_size,
                inference_window_stride=inference_window_stride,
            )
        else:
            logits = self._forward_encoder_window(audio, attention_mask=attention_mask)

        input_lengths = (audio_length / sample_rate * 50).ceil().long()

        if text_tokens is None:
            text_tokens = torch.tensor(
                [self.tokenizer.encode(t, add_special_tokens=False) for t in text], device=audio.device
            )
        token_positions, token_masks = self._align_text_tokens(logits, text_tokens, input_lengths)
        return AlignOutput(
            token_positions=token_positions, token_masks=token_masks, logits=logits if return_logits else None
        )

    @torch.no_grad()
    def _align_text_tokens(
        self,
        logits: torch.Tensor,
        text_tokens: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Align text tokens to the logits.

        Args:
            logits: The logits from the encoder.
            text_tokens: The text tokens to align to.
            input_lengths: The input lengths of the logits.
            batch_keys: Optional list of sample keys for caching.

        Returns:
            all_selected_positions: has the same length as text_tokens, each element indicates the position of the text
                token in the logits.
            text_token_mask: has the same length as the logits length, each element is 0 or 1, indicating whether the position
                is used for the text token.
        """

        def process_single_item(_logits, _text_tokens, _input_length):
            """Process a single item in the batch."""
            # Always compute alignment (no cache reading)
            valid_tokens = torch.nn.functional.pad(_text_tokens, (1, 0))
            new_logits = torch.ones_like(_logits, device="cpu", dtype=torch.float32) * -float("inf")
            # Modify the original logits tensor directly instead of the view
            new_logits[:, valid_tokens] = _logits[:, valid_tokens].float().cpu()
            _text_tokens_cpu = _text_tokens.cpu()
            try:
                selected_positions = _align_text_tokens(
                    new_logits, _text_tokens_cpu[_text_tokens_cpu != self.tokenizer.eos_token_id]
                )
            except Exception as e:
                raise Exception(f"Error aligning text tokens: {e}")
            pos_emb = torch.zeros(input_lengths.max(), dtype=torch.long, device=logits.device)
            pos_emb[selected_positions] = 1
            selected_positions_tensor = 1 + torch.tensor(selected_positions, dtype=torch.long, device=logits.device)

            return selected_positions_tensor, pos_emb

        results = [
            process_single_item(_logits, _text_tokens, _input_lengths.item())
            for _logits, _text_tokens, _input_lengths in zip(logits, text_tokens, input_lengths)
        ]

        # Unpack results and write to cache
        all_token_positions = []
        all_token_masks = []

        for i, (selected_pos, mask) in enumerate(results):
            all_token_positions.append(selected_pos)
            all_token_masks.append(mask)

        all_token_masks = torch.stack(all_token_masks, dim=0)
        all_token_positions = torch.nn.utils.rnn.pad_sequence(all_token_positions, batch_first=True, padding_value=0)
        return all_token_positions, all_token_masks
