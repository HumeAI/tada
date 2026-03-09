import math
import time
from dataclasses import dataclass, replace
from typing import Literal, Optional

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast as CausalLMOutputWithPastBase
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils.generic import ModelOutput

from ..nn.vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig
from ..utils.gray_code import decode_gray_code_to_time
from ..utils.text import normalize_text as normalize_text_fn
from .acoustic_spkr_verf import AcousticSpkrVerf
from .decoder import Decoder, DecoderConfig
from .encoder import Encoder, EncoderConfig, EncoderOutput


@dataclass
class InferenceOptions:
    text_do_sample: bool = True
    text_temperature: float = 0.6
    text_top_k: int = 0
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.1
    acoustic_cfg_scale: float = 1.6
    duration_cfg_scale: float = 1.0
    cfg_schedule: Literal["constant", "linear", "cosine"] = "cosine"
    noise_temperature: float = 0.9
    num_flow_matching_steps: int = 20
    time_schedule: Literal["uniform", "cosine", "logsnr"] = "logsnr"
    num_acoustic_candidates: int = 1
    scorer: Literal["spkr_verification", "likelihood", "duration_median"] = "likelihood"
    spkr_verification_weight: float = 1.0
    speed_up_factor: float | None = None
    negative_condition_source: Literal["negative_step_output", "prompt", "zero"] = "negative_step_output"
    text_only_logit_scale: float = 0.0


class TadaConfig(LlamaConfig):
    def __init__(
        self,
        acoustic_dim: int = 512,
        num_time_classes: int = 1024,
        latent_dropout: float = 0.0,
        add_semantic_to_condition: float = 0.0,
        shift_acoustic: int = 5,
        acoustic_from_nth_hidden_state: int = -1,
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        dist_type: str = "fixed",
        diffusion_head_type: str = "vibevoice",
        bottleneck_dim: int | None = None,
        context_window: int = 8,
        acoustic_mean: float = 0.0,
        acoustic_std: float = 1.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.acoustic_dim = acoustic_dim

        self.num_time_classes = num_time_classes
        self.latent_dropout = latent_dropout
        self.add_semantic_to_condition = add_semantic_to_condition
        self.shift_acoustic = shift_acoustic
        self.acoustic_from_nth_hidden_state = acoustic_from_nth_hidden_state
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.dist_type = dist_type
        self.diffusion_head_type = diffusion_head_type
        self.bottleneck_dim = bottleneck_dim
        self.context_window = context_window
        self.acoustic_mean = acoustic_mean
        self.acoustic_std = acoustic_std


@dataclass
class SyncTokGenerationOutput(ModelOutput):
    """
    Output type for sync token generation.

    Args:
        acoustic_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, acoustic_dim)`):
            The generated acoustic features.
    """

    acoustic_features: torch.FloatTensor | None = None
    time_before: torch.LongTensor | None = None
    past_key_values: Optional[torch.Tensor] | None = None
    text_token_ids: torch.LongTensor | None = None
    llm_time: torch.FloatTensor | None = None
    diffusion_time: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    step_logs: list[dict] | None = None
    text: list[str] | None = None


@dataclass
class GenerationOutput(ModelOutput):
    """
    Output type for generation.
    """

    audio: list[torch.Tensor] | None = None
    text: list[str] | None = None
    acoustic_features: torch.FloatTensor | None = None
    time_before: torch.LongTensor | None = None
    past_key_values: Optional[torch.Tensor] = None
    prompt_text_tokens: torch.LongTensor | None = None
    text_token_ids: torch.LongTensor | None = None
    output_str: list[str] | None = None
    output_text_ids: torch.LongTensor | None = None
    input_str: list[str] | None = None
    input_text_ids: torch.LongTensor | None = None
    llm_time: torch.FloatTensor | None = None
    diffusion_time: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    step_logs: list[dict] | None = None


class CausalLMOutputWithPast(CausalLMOutputWithPastBase):
    def __init__(
        self,
        ce_loss: Optional[torch.FloatTensor] = None,
        diffusion_loss: Optional[torch.FloatTensor] = None,
        time_loss: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batched_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ce_loss = ce_loss
        self.diffusion_loss = diffusion_loss
        self.time_loss = time_loss
        self.inputs_embeds = inputs_embeds
        self.batched_input_ids = batched_input_ids


class TadaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: TadaConfig):
        super().__init__(config)

        # Calculate number of bits for gray code early (needed for dimension calculations)
        self.num_time_bits = math.ceil(math.log2(config.num_time_classes))

        # Calculate additional dimensions for time conditioning
        self.time_dim = time_dim = 2 * self.num_time_bits  # 2 sets of gray code bits

        if config.diffusion_head_type == "vibevoice":
            self.prediction_head = VibeVoiceDiffusionHead(
                VibeVoiceDiffusionHeadConfig(
                    diffusion_type="ddpm",
                    head_ffn_ratio=config.head_ffn_ratio,
                    head_layers=config.head_layers,
                    hidden_size=self.config.hidden_size if config.bottleneck_dim is None else config.bottleneck_dim,
                    latent_size=self.config.acoustic_dim + time_dim,
                    model_type="vibevoice_diffusion_head",
                    rms_norm_eps=1e-05,
                    speech_vae_dim=self.config.acoustic_dim + time_dim,
                )
            )
            self.bottleneck_proj = (
                torch.nn.Linear(config.hidden_size, config.bottleneck_dim)
                if config.bottleneck_dim is not None
                else torch.nn.Identity()
            )

        self.acoustic_proj = torch.nn.Linear(config.acoustic_dim, config.hidden_size)
        self.time_start_embed = torch.nn.Embedding(config.num_time_classes, config.hidden_size)
        self.time_end_embed = torch.nn.Embedding(config.num_time_classes, config.hidden_size)
        self.acoustic_mask_emb = torch.nn.Embedding(num_embeddings=2, embedding_dim=config.hidden_size)
        self.acoustic_mask_emb.weight.data.fill_(0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        self = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        self._encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder")
        self._decoder = Decoder.from_pretrained("HumeAI/tada-codec", subfolder="decoder")
        return self

    @property
    def encoder(self) -> Encoder:
        if not hasattr(self, "_encoder"):
            self._encoder = Encoder(EncoderConfig())
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        if not hasattr(self, "_decoder"):
            self._decoder = Decoder(DecoderConfig())
        return self._decoder

    @property
    def acoustic_spkr_verf(self) -> AcousticSpkrVerf | None:
        if not hasattr(self, "_acoustic_spkr_verf"):
            self._acoustic_spkr_verf = None
        return self._acoustic_spkr_verf

    def load_acoustic_spkr_verf(self, model_name: str = "HumeAI/tada-codec", subfolder: str = "spkr-verf") -> AcousticSpkrVerf:
        self._acoustic_spkr_verf = AcousticSpkrVerf.from_pretrained(model_name, subfolder=subfolder)
        self._acoustic_spkr_verf.to(self.device)
        self._acoustic_spkr_verf.eval()
        return self._acoustic_spkr_verf

    def _lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run lm_head, falling back to CPU for MPS (output channels >65536 unsupported)."""
        if hidden_states.device.type == "mps":
            return self.lm_head(hidden_states.to("cpu")).to(hidden_states.device)
        return self.lm_head(hidden_states)

    def _build_prompt_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        prompt_acoustic_features: torch.FloatTensor | None,
        prompt_acoustic_masks: torch.LongTensor | None,
        prompt_time_len_before: torch.LongTensor | None,
        prompt_time_len_after: torch.LongTensor | None,
        prompt_len: int,
    ) -> torch.Tensor:
        """Build inputs_embeds for positions 0..prompt_len-1 for prefill. Returns (batch, prompt_len, hidden_size).

        Must match step-by-step loop: at step t we build embedding for position t using (acoustic, time)
        set at step t-1. So position t<=shift gets (0,0); position t>shift gets
        acoustic=prompt_acoustic_features[:, t-shift-1], time=prompt_time_len_before[:, t-shift].
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        shift = self.config.shift_acoustic

        token_emb = self.model.embed_tokens(input_ids[:, :prompt_len])

        # Acoustic: position t>shift uses prompt[:, t-shift-1] (value set at step t-1 in step-by-step)
        acoustic_full = torch.zeros(
            batch_size, prompt_len, self.config.acoustic_dim, device=device, dtype=token_emb.dtype
        )
        if prompt_acoustic_features is not None and prompt_acoustic_masks is not None:
            n_ac = min(prompt_len - shift - 1, prompt_acoustic_features.shape[1])
            if n_ac > 0:
                acoustic_full[:, shift + 1 : shift + 1 + n_ac] = prompt_acoustic_features[:, :n_ac]
        masks_full = torch.zeros(batch_size, prompt_len, device=device, dtype=torch.long)
        if prompt_acoustic_masks is not None:
            n_ac = min(prompt_len - shift - 1, prompt_acoustic_masks.shape[1])
            if n_ac > 0:
                masks_full[:, shift + 1 : shift + 1 + n_ac] = prompt_acoustic_masks[:, :n_ac]
        acoustic_emb = self.acoustic_proj(acoustic_full) + self.acoustic_mask_emb(masks_full)

        # Time: position t>shift uses prompt_time_*[:, t-shift] (set at step t-1 in step-by-step).
        # Before/after use the same column index (time_after is already offset in generate(): time_gaps[:,1:]).
        time_before = torch.zeros(batch_size, prompt_len, device=device, dtype=torch.long)
        time_after = torch.zeros(batch_size, prompt_len, device=device, dtype=torch.long)
        if prompt_time_len_before is not None and prompt_time_len_after is not None:
            # Indices 1..(prompt_len-shift-1) for positions shift+1..prompt_len-1 (last prompt frame included when shape[1] >= prompt_len-shift).
            n_t = min(prompt_len - shift - 1, prompt_time_len_before.shape[1] - 1)
            if n_t > 0:
                time_before[:, shift + 1 : shift + 1 + n_t] = prompt_time_len_before[:, 1 : 1 + n_t]
                time_after[:, shift + 1 : shift + 1 + n_t] = prompt_time_len_after[:, 1 : 1 + n_t]
        time_emb = self.time_start_embed(time_before) + self.time_end_embed(time_after)

        return token_emb + acoustic_emb + time_emb

    def forward_one_step(
        self,
        input_ids: torch.LongTensor,
        acoustic_features: torch.FloatTensor,
        acoustic_masks: torch.LongTensor,
        time_len_before: torch.LongTensor,
        time_len_after: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        compute_logits: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = (
            self.model.embed_tokens(input_ids)
            + self.acoustic_proj(acoustic_features)
            + self.acoustic_mask_emb(acoustic_masks.long())
            + self.time_start_embed(time_len_before)
            + self.time_end_embed(time_len_after)
        )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=False,
            **kwargs,
        )

        last_hidden = outputs.last_hidden_state
        logits = self._lm_head_forward(last_hidden) if compute_logits else None

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(last_hidden,),
        )

    def _compute_velocity(
        self,
        speech_input: torch.Tensor,
        t: torch.Tensor,
        cond_input: torch.Tensor,
        neg_cond_input: torch.Tensor,
        acoustic_cfg: float,
        duration_cfg: float,
    ) -> torch.Tensor:
        """
        Compute velocity for flow matching with optional classifier-free guidance.

        Args:
            speech_input: Current speech state
            t: Current timestep
            cond_input: Conditioning hidden states
            acoustic_cfg: CFG scale for acoustic features
            duration_cfg: CFG scale for duration features

        Returns:
            Predicted velocity
        """
        if acoustic_cfg != 1.0:
            # Duplicate speech input for both conditional and unconditional
            speech_combined = torch.cat([speech_input, speech_input], dim=0)
            t_combined = t.repeat(speech_input.shape[0] * 2).to(speech_input)

            # Positive condition: actual condition
            # Negative condition: zeros (unconditional)
            cond_pos = cond_input.squeeze(1)  # (batch, hidden)
            cond_neg = neg_cond_input.squeeze(1)  # (batch, hidden)
            cond_combined = torch.cat([cond_pos, cond_neg], dim=0)

            # Forward pass with both conditions
            velocity_combined = self.prediction_head(
                speech_combined, t_combined, condition=self.bottleneck_proj(cond_combined)
            )

            # Split and apply CFG formula
            velocity_pos, velocity_neg = torch.chunk(velocity_combined, 2, dim=0)
            velocity = torch.cat(
                [
                    (velocity_neg + acoustic_cfg * (velocity_pos - velocity_neg))[..., : self.config.acoustic_dim],
                    (velocity_neg + duration_cfg * (velocity_pos - velocity_neg))[..., self.config.acoustic_dim :],
                ],
                dim=-1,
            )
        else:
            # No CFG, standard forward pass
            velocity = self.prediction_head(
                speech_input,
                t.repeat(speech_input.shape[0]).to(speech_input),
                condition=self.bottleneck_proj(cond_input.squeeze(1)),
            )
        return velocity

    @staticmethod
    def _scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
        """Compute effective CFG scale at timestep t given a schedule.

        Args:
            base_scale: The target CFG scale (e.g. 1.3).
            t: Current ODE timestep in [0, 1].
            schedule: One of "constant", "linear", "cosine".
                - constant: base_scale at all t.
                - linear:   linearly decays from base_scale at t=0 to 1.0 at t=1.
                - cosine:   cosine decay from base_scale at t=0 to 1.0 at t=1 (smoother).

        Returns:
            Effective CFG scale.
        """
        if schedule == "constant" or base_scale == 1.0:
            return base_scale
        if schedule == "linear":
            return 1.0 + (base_scale - 1.0) * (1.0 - t)
        if schedule == "cosine":
            return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
        return base_scale

    @staticmethod
    def _build_time_schedule(num_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
        """Build a time schedule for ODE discretization.

        Args:
            num_steps: Number of ODE steps.
            schedule: One of "uniform", "cosine", "logsnr".
                - uniform: evenly spaced in [0, 1] (original behavior).
                - cosine:  denser near t=0 and t=1 where velocity changes fastest.
                - logsnr:  spaced uniformly in log-SNR space, concentrating steps
                           near t=0 (denoising onset) where accuracy matters most.

        Returns:
            Tensor of shape (num_steps + 1,) with values in [0, 1].
        """
        if schedule == "cosine":
            # t = 0.5 * (1 - cos(π * u))  where u is uniform in [0, 1]
            u = torch.linspace(0, 1, num_steps + 1, device=device)
            return 0.5 * (1 - torch.cos(math.pi * u))
        if schedule == "logsnr":
            # Uniform in log-SNR: SNR(t) = (1-t)²/t² for linear interpolation.
            # log_snr goes from +∞ at t=0 to -∞ at t=1. We clip to a practical range
            # and map back to t.  t = σ(−log_snr/2) where σ is the sigmoid.
            log_snr = torch.linspace(5.0, -5.0, num_steps + 1, device=device)
            t_span = torch.sigmoid(-log_snr / 2)
            # Ensure exact endpoints
            t_span[0] = 0.0
            t_span[-1] = 1.0
            return t_span
        # Default: uniform
        return torch.linspace(0, 1, num_steps + 1, device=device)

    def _solve_flow_matching(
        self,
        speech: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        num_steps: int,
        acoustic_cfg_scale: float,
        duration_cfg_scale: float,
        cfg_schedule: str = "constant",
        time_schedule: str = "uniform",
        forced_time_before: torch.Tensor | None = None,
        forced_time_after: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Solve the flow matching ODE to generate speech features using Euler method.

        Args:
            speech: Initial noise state (batch, feature_dim)
            cond: Conditioning hidden states (batch, 1, hidden_dim)
            num_steps: Number of discretization steps
            acoustic_cfg_scale: Peak CFG scale for acoustic features
            duration_cfg_scale: Peak CFG scale for duration features
            cfg_schedule: How CFG scale varies over ODE timesteps.
                "constant" — fixed at the given scale (default, original behavior).
                "linear"   — linearly decays from scale at t=0 to 1.0 at t=1.
                "cosine"   — cosine decay from scale at t=0 to 1.0 at t=1.
            time_schedule: How ODE timesteps are distributed in [0, 1].
                "uniform" — evenly spaced (original behavior).
                "cosine"  — denser near t=0 and t=1.
                "logsnr"  — uniform in log-SNR space, denser near t=0.
            forced_time_before: Optional gray code bit vector for time_before (batch, num_time_bits).
                When provided, replaces time_before slots with the noised interpolant at each step.
            forced_time_after: Optional gray code bit vector for time_after (batch, num_time_bits).
                When provided, replaces time_after slots with the noised interpolant at each step.

        Returns:
            Final speech state after ODE solving
        """
        t_span = self._build_time_schedule(num_steps, time_schedule, cond.device)
        t_curr = t_span[0]

        has_forced_time = forced_time_before is not None or forced_time_after is not None
        if has_forced_time:
            acoustic_dim = self.config.acoustic_dim
            if forced_time_before is not None:
                time_noise_before = speech[..., acoustic_dim : acoustic_dim + self.num_time_bits].clone()
            if forced_time_after is not None:
                time_noise_after = speech[..., acoustic_dim + self.num_time_bits :].clone()

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_curr
            t_val = t_curr.item()
            a_cfg = self._scheduled_cfg(acoustic_cfg_scale, t_val, cfg_schedule)
            d_cfg = self._scheduled_cfg(duration_cfg_scale, t_val, cfg_schedule)

            velocity = self._compute_velocity(speech, t_curr, cond, neg_cond, a_cfg, d_cfg)
            speech = speech + dt * velocity

            if has_forced_time:
                t_next = t_span[i]
                if forced_time_before is not None:
                    speech[..., acoustic_dim : acoustic_dim + self.num_time_bits] = (
                        1 - t_next
                    ) * time_noise_before + t_next * forced_time_before
                if forced_time_after is not None:
                    speech[..., acoustic_dim + self.num_time_bits :] = (
                        1 - t_next
                    ) * time_noise_after + t_next * forced_time_after

            t_curr = t_span[i]

        return speech

    def _score_by_reconstruction(
        self,
        samples: torch.Tensor,
        noise: torch.Tensor,
        cond: torch.Tensor,
        num_eval_points: int = 3,
    ) -> torch.Tensor:
        """
        Score samples by how well the conditional model (no CFG) predicts the
        OT velocity along the straight-line path from noise to sample.

        For each candidate, evaluates the flow matching reconstruction loss at
        multiple timesteps along the OT interpolant x_t = (1-t)*noise + t*sample.
        Lower loss means the model "agrees" this (noise, sample) pair is on-distribution.

        Uses the conditional velocity (CFG=1.0) because:
        - This matches the training objective (flow matching loss)
        - The CFG-modified velocity doesn't correspond to the learned model

        All timesteps are batched into a single forward pass for efficiency.
        Eval points are concentrated in [0.3, 0.9] where the model is most
        discriminative (near t=0, x_t ≈ noise for all candidates, giving
        little signal to distinguish quality).

        Args:
            samples: Generated samples x_1, shape (batch, dim)
            noise: Initial noise x_0 that produced the samples, shape (batch, dim)
            cond: Conditioning hidden states (batch, 1, hidden_dim)
            num_eval_points: Number of timesteps to evaluate at (default 3)

        Returns:
            scores: Negative reconstruction loss for each sample, shape (batch,).
                    Higher is better.
        """
        batch = samples.shape[0]
        acoustic_dim = self.config.acoustic_dim
        target_velocity = samples - noise  # OT velocity (constant along straight line)

        # Concentrate eval points in mid-to-late range where model is most discriminative
        t_points = torch.linspace(0.3, 0.9, num_eval_points, device=samples.device)

        # Batch all timesteps into a single forward pass instead of looping
        x_t_all = torch.cat([(1 - t) * noise + t * samples for t in t_points], dim=0)
        t_all = torch.cat([t.expand(batch) for t in t_points], dim=0)
        cond_proj = self.bottleneck_proj(cond.squeeze(1))  # compute once, reuse
        cond_all = cond_proj.repeat(num_eval_points, 1)

        pred_velocity_all = self.prediction_head(x_t_all, t_all, condition=cond_all)

        # Compute per-sample MSE on acoustic dims only, then average across timesteps
        target_all = target_velocity.repeat(num_eval_points, 1)
        error = (pred_velocity_all - target_all)[..., :acoustic_dim].pow(2).mean(-1)

        return -error.view(num_eval_points, batch).mean(dim=0)

    def _solve_flow_matching_ranked(
        self,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        opts: InferenceOptions,
        ref_spkr_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate multiple flow matching candidates and select the best one.

        Args:
            cond: Conditioning hidden states (batch, 1, hidden_dim)
            opts: Inference options containing scorer, num_acoustic_candidates, etc.
            ref_spkr_emb: Mean speaker embedding from prompt frames for spkr_verification scorer (batch, 192)

        Returns:
            Best speech candidate (batch, total_dim)
        """
        batch_size = cond.shape[0]
        total_dim = self.config.acoustic_dim + self.time_dim
        num_candidates = opts.num_acoustic_candidates

        # Sample N different initial noises (scaled by noise temperature)
        noise = (
            torch.randn(num_candidates * batch_size, total_dim, device=cond.device, dtype=cond.dtype)
            * opts.noise_temperature
        )
        cond_expanded = cond.repeat(num_candidates, 1, 1)
        # Use actual neg_cond from generation loop (respects negative_condition_source)
        if neg_cond.dim() == 3:
            neg_cond_expanded = neg_cond.repeat(num_candidates, 1, 1)
        elif neg_cond.dim() == 2:
            neg_cond_expanded = neg_cond.unsqueeze(1).repeat(num_candidates, 1, 1)
        else:
            neg_cond_expanded = torch.zeros_like(cond_expanded)

        # Generate candidates with CFG
        speech_flat = self._solve_flow_matching(
            speech=noise.clone(),
            cond=cond_expanded,
            neg_cond=neg_cond_expanded,
            num_steps=opts.num_flow_matching_steps,
            acoustic_cfg_scale=opts.acoustic_cfg_scale,
            duration_cfg_scale=opts.duration_cfg_scale,
            cfg_schedule=opts.cfg_schedule,
            time_schedule=opts.time_schedule,
        )

        speech_candidates = speech_flat.view(num_candidates, batch_size, total_dim)

        if opts.scorer == "spkr_verification" and self.acoustic_spkr_verf is not None:
            # Rank by speaker verification: cosine sim between candidate and mean prompt speaker embedding
            if ref_spkr_emb is not None:
                acoustic_only = speech_candidates[..., : self.config.acoustic_dim]
                acoustic_only_norm = (acoustic_only - self.config.acoustic_mean) / self.config.acoustic_std

                with torch.no_grad():
                    cand_spkr_emb = self.acoustic_spkr_verf(
                        acoustic_only_norm.reshape(-1, self.config.acoustic_dim)
                    ).view(num_candidates, batch_size, -1)  # [N, B, 192]

                # Cosine similarity to mean prompt speaker embedding (global anchor)
                scores = opts.spkr_verification_weight * torch.einsum("nbe,be->nb", cand_spkr_emb, ref_spkr_emb)
                best_idx = scores.argmax(dim=0)
            else:
                best_idx = torch.zeros(batch_size, device=cond.device, dtype=torch.long)
        elif opts.scorer == "duration_median":
            # Rank by duration: pick the candidate closest to the median duration
            time_gray = speech_candidates[..., -self.time_dim :]
            cand_time_before = decode_gray_code_to_time(
                time_gray[..., : self.num_time_bits], self.num_time_bits
            )  # (N, B)
            cand_time_after = decode_gray_code_to_time(
                time_gray[..., self.num_time_bits :], self.num_time_bits
            )  # (N, B)
            cand_duration = cand_time_before + cand_time_after  # (N, B)
            median_duration = cand_duration.median(dim=0).values  # (B,)
            best_idx = (cand_duration - median_duration.unsqueeze(0)).abs().argmin(dim=0)
        else:
            # Rank by conditional reconstruction loss (no CFG, no grad needed)
            scores = self._score_by_reconstruction(
                samples=speech_flat,
                noise=noise,
                cond=cond_expanded,
            )
            best_idx = scores.view(num_candidates, batch_size).argmax(dim=0)

        batch_indices = torch.arange(batch_size, device=cond.device)
        return speech_candidates[best_idx, batch_indices]

    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        prompt_acoustic_features: torch.FloatTensor | None = None,
        prompt_acoustic_masks: torch.LongTensor | None = None,
        prompt_time_len_before: torch.LongTensor | None = None,
        prompt_time_len_after: torch.LongTensor | None = None,
        num_steps: int = 1024,
        log_time: bool = True,
        inference_options: InferenceOptions = InferenceOptions(),
        use_text_in_prompt: bool = False,
        verbose: bool = False,
        return_logits: bool = False,
        **kwargs,
    ) -> SyncTokGenerationOutput:
        start_header_id = self.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if not use_text_in_prompt:
            # Replace text content tokens with pad only within the prompt region,
            # keeping structural tokens (start_header_id, end_header_id, eot_id,
            # tokens between start/end header, bos, eos).
            prompt_token_len = prompt_acoustic_features.shape[1] if prompt_acoustic_features is not None else 0
            pad_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            keep_mask = torch.zeros(input_ids.shape[0], prompt_token_len, dtype=torch.bool, device=input_ids.device)
            for b in range(input_ids.shape[0]):
                in_header = False
                for t in range(prompt_token_len):
                    token = input_ids[b, t].item()
                    if token == start_header_id:
                        in_header = True
                        keep_mask[b, t] = True
                    elif token == end_header_id:
                        in_header = False
                        keep_mask[b, t] = True
                    elif in_header:
                        keep_mask[b, t] = True
                    elif token in (eot_id, bos_id, eos_id):
                        keep_mask[b, t] = True
            input_ids = input_ids.clone()
            input_ids[:, :prompt_token_len][~keep_mask] = pad_id

        if verbose:
            print("Prompt:", self.tokenizer.decode(input_ids[0]))

        opts = inference_options
        acoustic_features = torch.zeros(input_ids.shape[0], 1, self.config.acoustic_dim, device=input_ids.device)
        acoustic_masks = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
        time_len_before = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
        time_len_after = torch.zeros(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)

        generation_config = GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, True)
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, 1, num_steps)
        model_kwargs["cache_position"] = torch.arange(1, device=input_ids.device, dtype=torch.long)

        all_acoustic_features: list[torch.FloatTensor] = []
        all_time_before: list[torch.LongTensor] = []
        all_logits: list[torch.FloatTensor] = [] if return_logits else None
        all_output_token_ids: list[torch.LongTensor] = []
        llm_time: list[float] = []
        diffusion_time: list[float] = []
        if log_time and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        acoustic_feat_type = "none"
        time_len_type = "none"

        prompt_len = input_ids.shape[1]
        step_start = 0
        shift_acoustic = self.config.shift_acoustic
        neg_cond = torch.zeros(
            input_ids.shape[0], self.config.hidden_size, device=input_ids.device, dtype=input_ids.dtype
        )
        prefill_conditions = None
        pad_token_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        use_neg_sampling = opts.acoustic_cfg_scale != 1.0
        # Only run a double (pos+neg) batch when the negative condition comes from the step output
        need_neg_batch = use_neg_sampling and opts.negative_condition_source == "negative_step_output"
        use_text_only_logit_scale = opts.text_only_logit_scale > 0.0

        if (
            prompt_len > 0
            and prompt_acoustic_features is not None
            and prompt_acoustic_masks is not None
            and prompt_time_len_before is not None
            and prompt_time_len_after is not None
        ):
            n_ac = min(prompt_len - shift_acoustic - 1, prompt_acoustic_features.shape[1] - 1)
            n_t = min(prompt_len - shift_acoustic - 1, prompt_time_len_before.shape[1] - 1)
            # Need n_prefill_frames <= shape[1]-1 so prompt_time_len_before[:, 1..n_prefill_frames] and [:, n_prefill_frames] are valid.
            n_frames_cap = max(0, prompt_time_len_before.shape[1] - 2)
            n_prefill_frames_max = min(n_ac, n_t, n_frames_cap) if (n_ac > 0 and n_t > 0) else 0
            prefill_len = min(prompt_len, shift_acoustic + n_prefill_frames_max + 1) if n_prefill_frames_max > 0 else 0
        else:
            prefill_len = 0

        B = input_ids.shape[0]

        if prefill_len > 0:
            if log_time:
                if torch.cuda.is_available():
                    start_event.record()
                else:
                    start_time = time.time()
            inputs_embeds_prefill = self._build_prompt_inputs_embeds(
                input_ids,
                prompt_acoustic_features,
                prompt_acoustic_masks,
                prompt_time_len_before,
                prompt_time_len_after,
                prefill_len,
            )

            # Batch pos+neg prefill into a single forward pass
            if need_neg_batch:
                combined_embeds = torch.cat([inputs_embeds_prefill, inputs_embeds_prefill], dim=0)
            else:
                combined_embeds = inputs_embeds_prefill
            if use_text_only_logit_scale:
                device = input_ids.device
                dtype = inputs_embeds_prefill.dtype
                text_only_prefill = (
                    self.model.embed_tokens(input_ids[:, :prefill_len])
                    + self.acoustic_proj(
                        torch.zeros(B, prefill_len, self.config.acoustic_dim, device=device, dtype=dtype)
                    )
                    + self.acoustic_mask_emb(torch.zeros(B, prefill_len, device=device, dtype=torch.long))
                    + self.time_start_embed(torch.zeros(B, prefill_len, device=device, dtype=torch.long))
                    + self.time_end_embed(torch.zeros(B, prefill_len, device=device, dtype=torch.long))
                )
                combined_embeds = torch.cat([combined_embeds, text_only_prefill], dim=0)

            prefill_outputs = self.model(
                inputs_embeds=combined_embeds,
                use_cache=True,
                past_key_values=None,
                cache_position=torch.arange(prefill_len, device=input_ids.device),
                output_hidden_states=False,
            )
            if log_time:
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    llm_time.append(start_event.elapsed_time(end_event))
                else:
                    end_time = time.time()
                    llm_time.append(end_time - start_time)
            model_kwargs["past_key_values"] = prefill_outputs.past_key_values
            pos_hidden = prefill_outputs.last_hidden_state[:B]
            if opts.negative_condition_source == "prompt":
                prefill_conditions = pos_hidden
            if all_logits is not None:
                prefill_logits = self._lm_head_forward(pos_hidden)
                if use_text_only_logit_scale:
                    text_only_hidden = prefill_outputs.last_hidden_state[-B:]
                    text_only_prefill_logits = self._lm_head_forward(text_only_hidden)
                    scale = opts.text_only_logit_scale
                    prefill_logits = (text_only_prefill_logits * scale + prefill_logits) / (scale + 1)
                for s in range(prefill_len):
                    all_logits.append(prefill_logits[:, s : s + 1])
            for s in range(prefill_len - 1):
                all_output_token_ids.append(input_ids[:, s + 1 : s + 2])

            n_prefill_frames = prefill_len - shift_acoustic
            for i in range(n_prefill_frames):
                all_acoustic_features.append(prompt_acoustic_features[:, i].unsqueeze(1))
            for i in range(n_prefill_frames):
                all_time_before.append(prompt_time_len_before[:, i + 1].unsqueeze(1))
            acoustic_features = prompt_acoustic_features[:, n_prefill_frames - 1].unsqueeze(1)
            acoustic_masks = prompt_acoustic_masks[:, n_prefill_frames - 1].unsqueeze(1)
            time_len_before = prompt_time_len_before[:, n_prefill_frames].unsqueeze(1)
            time_len_after = prompt_time_len_after[:, n_prefill_frames].unsqueeze(1)
            acoustic_feat_type = "prompted"
            time_len_type = "prompted"

            model_kwargs["cache_position"] = torch.tensor([prefill_len], device=input_ids.device, dtype=torch.long)
            step_start = prefill_len

        # Auto-load scorer models if needed
        if opts.num_acoustic_candidates > 1:
            if opts.scorer == "spkr_verification" and self.acoustic_spkr_verf is None:
                self.load_acoustic_spkr_verf()

        # Pre-compute reference speaker embedding from prompt frames (for spkr_verification scorer)
        ref_spkr_emb = None
        if (
            self.acoustic_spkr_verf is not None
            and prompt_acoustic_features is not None
            and opts.num_acoustic_candidates > 1
            and opts.scorer == "spkr_verification"
        ):
            prompt_norms = prompt_acoustic_features.norm(dim=-1)  # [B, T]
            valid_mask_sv = prompt_norms > 0  # [B, T]
            with torch.no_grad():
                pf_norm_sv = (prompt_acoustic_features - self.config.acoustic_mean) / self.config.acoustic_std
                B_sv, T_sv, D_sv = pf_norm_sv.shape
                all_spkr_embs = self.acoustic_spkr_verf(pf_norm_sv.reshape(-1, D_sv)).view(B_sv, T_sv, -1)
                valid_mask_sv_exp = valid_mask_sv.unsqueeze(-1).float()  # [B, T, 1]
                # Mean-pool speaker embeddings over valid frames, then re-normalize
                ref_spkr_emb = (all_spkr_embs * valid_mask_sv_exp).sum(dim=1) / valid_mask_sv_exp.sum(dim=1).clamp(
                    min=1
                )
                ref_spkr_emb = torch.nn.functional.normalize(ref_spkr_emb, dim=-1)  # [B, 192]

        step_logs = []
        # Add step_log entries for prefilled steps (so alignment visualization is complete)
        for s in range(step_start):
            token_id = input_ids[0, s].item()
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            if s >= shift_acoustic and (s - shift_acoustic + 1) < prompt_time_len_before.shape[1]:
                t_before = prompt_time_len_before[0, s - shift_acoustic + 1].item()
                t_after = prompt_time_len_after[0, s - shift_acoustic + 1].item()
            else:
                t_before = 0
                t_after = 0
            step_logs.append({
                "step": s,
                "token": token_str,
                "n_frames_before": t_before,
                "n_frames_after": t_after,
                "n_frames_src": "prompted",
                "acoustic_mask": 1 if s >= shift_acoustic else 0,
                "acoustic_feat_src": "prefilled",
                "acoustic_feat_norm": 0.0,
            })
        last_time_before = None
        for step in range(step_start, num_steps):
            # When step >= input_ids.shape[1] we are generating; use last token as input for forward
            input_slice = input_ids[:, step : step + 1] if step < input_ids.shape[1] else input_ids[:, -1:]

            if log_time:
                if torch.cuda.is_available():
                    start_event.record()
                else:
                    start_time = time.time()
            step_logs.append(
                {
                    "step": step,
                    "token": self.tokenizer.convert_ids_to_tokens([input_slice[0, 0].item()])[0],
                    "n_frames_before": time_len_before[0].item(),
                    "n_frames_after": time_len_after[0].item(),
                    "n_frames_src": time_len_type,
                    "acoustic_mask": acoustic_masks[0].item(),
                    "acoustic_feat_src": acoustic_feat_type,
                    "acoustic_feat_norm": acoustic_features[0].norm().item(),
                }
            )

            need_logits = return_logits or step >= input_ids.shape[1] - 1

            if need_neg_batch:
                is_structural = (
                    (input_slice == start_header_id) | (input_slice == end_header_id) | (input_slice == eot_id)
                )
                neg_input_slice = torch.where(is_structural, input_slice, torch.full_like(input_slice, pad_token_id))
                combined_slice = torch.cat([input_slice, neg_input_slice], dim=0)
                neg_acoustic_features = acoustic_features
                combined_acoustic = torch.cat([acoustic_features, neg_acoustic_features], dim=0)
                combined_masks = torch.cat([acoustic_masks, acoustic_masks], dim=0)
                combined_time_before = torch.cat([time_len_before, time_len_before], dim=0)
                combined_time_after = torch.cat([time_len_after, time_len_after], dim=0)
                if use_text_only_logit_scale:
                    combined_slice = torch.cat([combined_slice, input_slice], dim=0)
                    combined_acoustic = torch.cat([combined_acoustic, torch.zeros_like(acoustic_features)], dim=0)
                    combined_masks = torch.cat([combined_masks, torch.zeros_like(acoustic_masks)], dim=0)
                    combined_time_before = torch.cat([combined_time_before, torch.zeros_like(time_len_before)], dim=0)
                    combined_time_after = torch.cat([combined_time_after, torch.zeros_like(time_len_after)], dim=0)
                model_inputs = self.prepare_inputs_for_generation(combined_slice, **model_kwargs)
                outputs = self.forward_one_step(
                    **model_inputs,
                    acoustic_features=combined_acoustic,
                    acoustic_masks=combined_masks,
                    time_len_before=combined_time_before,
                    time_len_after=combined_time_after,
                    compute_logits=need_logits,
                    **kwargs,
                )
                neg_cond = outputs.hidden_states[-1][B : 2 * B]
            else:
                if use_text_only_logit_scale:
                    combined_slice = torch.cat([input_slice, input_slice], dim=0)
                    combined_acoustic = torch.cat([acoustic_features, torch.zeros_like(acoustic_features)], dim=0)
                    combined_masks = torch.cat([acoustic_masks, torch.zeros_like(acoustic_masks)], dim=0)
                    combined_time_before = torch.cat([time_len_before, torch.zeros_like(time_len_before)], dim=0)
                    combined_time_after = torch.cat([time_len_after, torch.zeros_like(time_len_after)], dim=0)
                    model_inputs = self.prepare_inputs_for_generation(combined_slice, **model_kwargs)
                    outputs = self.forward_one_step(
                        **model_inputs,
                        acoustic_features=combined_acoustic,
                        acoustic_masks=combined_masks,
                        time_len_before=combined_time_before,
                        time_len_after=combined_time_after,
                        compute_logits=need_logits,
                        **kwargs,
                    )
                else:
                    model_inputs = self.prepare_inputs_for_generation(input_slice, **model_kwargs)
                    outputs = self.forward_one_step(
                        **model_inputs,
                        acoustic_features=acoustic_features,
                        acoustic_masks=acoustic_masks,
                        time_len_before=time_len_before,
                        time_len_after=time_len_after,
                        compute_logits=need_logits,
                        **kwargs,
                    )
                if opts.negative_condition_source == "prompt" and prefill_conditions is not None:
                    neg_cond = prefill_conditions[:, step % prefill_conditions.shape[1]].unsqueeze(1)

            if log_time:
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    llm_time.append(start_event.elapsed_time(end_event))
                else:
                    end_time = time.time()
                    llm_time.append(end_time - start_time)
            assert outputs.hidden_states is not None
            hidden_states = outputs.hidden_states[-1][:B]
            logits = outputs.logits[:B] if need_logits else None
            text_only_logits = outputs.logits[-B:] if (need_logits and use_text_only_logit_scale) else None

            cond = hidden_states

            if log_time:
                if torch.cuda.is_available():
                    start_event.record()
                else:
                    start_time = time.time()

            if opts.num_acoustic_candidates > 1:
                speech = self._solve_flow_matching_ranked(cond, neg_cond, opts, ref_spkr_emb=ref_spkr_emb)
            else:
                speech = torch.randn(cond.shape[0], self.config.acoustic_dim).to(cond) * opts.noise_temperature
                speech = torch.cat(
                    [speech, torch.randn(cond.shape[0], self.time_dim).to(cond) * opts.noise_temperature], dim=-1
                )

                # Solve flow matching ODE
                speech = self._solve_flow_matching(
                    speech=speech,
                    cond=cond,
                    neg_cond=neg_cond,
                    num_steps=opts.num_flow_matching_steps,
                    acoustic_cfg_scale=opts.acoustic_cfg_scale,
                    duration_cfg_scale=opts.duration_cfg_scale,
                    cfg_schedule=opts.cfg_schedule,
                    time_schedule=opts.time_schedule,
                )

            # Extract time_len_before and time_len_after from flow matching output
            time_len_gray_code = speech[..., -self.time_dim :]
            predicted_time_len_before = decode_gray_code_to_time(
                time_len_gray_code[..., : self.num_time_bits], self.num_time_bits
            ).unsqueeze(0)
            predicted_time_len_after = decode_gray_code_to_time(
                time_len_gray_code[..., self.num_time_bits :], self.num_time_bits
            ).unsqueeze(0)

            if all_logits is not None:
                all_logits.append(logits)
            if step >= input_ids.shape[1] - 1:
                token_logits = logits[:, -1, :].clone()
                # Prevent pad token from being generated
                token_logits[:, pad_token_id] = float("-inf")

                if text_only_logits is not None:
                    scale = opts.text_only_logit_scale
                    token_logits = (text_only_logits[:, -1, :] * scale + token_logits) / (scale + 1)

                if opts.text_do_sample:
                    # Repetition penalty: penalize tokens already present in the sequence
                    if opts.text_repetition_penalty != 1.0:
                        score = torch.gather(token_logits, 1, input_ids)
                        # If score < 0, multiply by penalty; if score > 0, divide by penalty
                        score = torch.where(
                            score < 0,
                            score * opts.text_repetition_penalty,
                            score / opts.text_repetition_penalty,
                        )
                        token_logits = token_logits.scatter(1, input_ids, score)

                    # Temperature scaling
                    token_logits = token_logits / opts.text_temperature

                    # Top-k filtering: keep only top k tokens
                    if opts.text_top_k > 0:
                        top_k = min(opts.text_top_k, token_logits.size(-1))
                        # Remove tokens with logits below the k-th largest
                        indices_to_remove = token_logits < torch.topk(token_logits, top_k, dim=-1).values[..., -1:]
                        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

                    # Top-p (nucleus) filtering: keep smallest set of tokens with cumulative prob >= top_p
                    if 0.0 < opts.text_top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        # Remove tokens with cumulative probability above the threshold
                        # Shift right so the first token exceeding threshold is kept
                        sorted_indices_to_remove = (
                            cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= opts.text_top_p
                        )
                        # Scatter back to original indices
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        token_logits = token_logits.masked_fill(indices_to_remove, float("-inf"))

                    probs = torch.softmax(token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = token_logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token.long()], dim=1)
                all_output_token_ids.append(next_token)
            else:
                all_output_token_ids.append(input_ids[:, step + 1].unsqueeze(1))

            if step >= self.config.shift_acoustic:
                if (
                    prompt_acoustic_features is not None
                    and step - self.config.shift_acoustic < prompt_acoustic_features.shape[1]
                ):
                    acoustic_features = prompt_acoustic_features[:, step - self.config.shift_acoustic].unsqueeze(1)
                    acoustic_feat_type = "prompted"
                    acoustic_masks = prompt_acoustic_masks[:, step - self.config.shift_acoustic].unsqueeze(1)
                else:
                    acoustic_features = speech.unsqueeze(0)
                    acoustic_feat_type = "predicted"
                    acoustic_masks = torch.ones(input_ids.shape[0], 1, device=input_ids.device, dtype=torch.long)
                    acoustic_features = (
                        acoustic_features[..., : -self.time_dim] if self.time_dim > 0 else acoustic_features
                    )
                all_acoustic_features.append(acoustic_features)

                if (
                    prompt_time_len_before is not None
                    and prompt_time_len_after is not None
                    and step - self.config.shift_acoustic < prompt_time_len_before.shape[1] - 1
                ):
                    time_len_type = "prompted"
                    time_len_before = prompt_time_len_before[:, step - self.config.shift_acoustic + 1].unsqueeze(1)
                    time_len_after = prompt_time_len_after[:, step - self.config.shift_acoustic + 1].unsqueeze(1)
                else:
                    # diffusion generating time length
                    time_len_type = "predicted"
                    time_len_before = predicted_time_len_before
                    time_len_after = predicted_time_len_after
                all_time_before.append(time_len_before)
                last_time_before = time_len_before
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if log_time:
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    diffusion_time.append(start_event.elapsed_time(end_event))
                else:
                    end_time = time.time()
                    diffusion_time.append(end_time - start_time)

        # _decode_wav needs len(time_before) == len(encoded) + 1; add trailing time from last step
        if last_time_before is not None:
            all_time_before.append(last_time_before)

        # If speed_up_factor != 1.0, rerun generation with scaled times from this pass
        if opts.speed_up_factor is not None:
            # Collect and scale all predicted times from this pass
            first_pass_time = torch.cat([t if t.ndim == 2 else t.unsqueeze(1) for t in all_time_before], dim=1)
            scaled_time = (first_pass_time.float() / opts.speed_up_factor).round().long()
            # scaled_time = first_pass_time.clone()
            # num_positions = scaled_time.shape[1]
            # num_to_scale = min(3, num_positions)
            # scale_indices = torch.randperm(num_positions, device=scaled_time.device)[:num_to_scale]
            # scaled_time[:, scale_indices] = (
            #     (scaled_time[:, scale_indices].float() / opts.speed_up_factor).round().long()
            # )

            # Build prompt_time tensors for second pass
            # Index 0 is unused padding; indices 1..N map to steps shift_acoustic..end
            second_pass_time_before = torch.cat([torch.zeros_like(scaled_time[:, :1]), scaled_time], dim=1)
            # time_after[i] = time_before[i+1], last position = 1
            second_pass_time_after = torch.cat([scaled_time, torch.ones_like(scaled_time[:, :1])], dim=1)

            second_pass_options = replace(inference_options, speed_up_factor=None)
            return self._generate(
                input_ids=input_ids,
                input_lengths=input_lengths,
                prompt_acoustic_features=prompt_acoustic_features,
                prompt_acoustic_masks=prompt_acoustic_masks,
                prompt_time_len_before=second_pass_time_before,
                prompt_time_len_after=second_pass_time_after,
                num_steps=input_ids.shape[1] - 1,
                log_time=log_time,
                inference_options=second_pass_options,
                verbose=verbose,
                return_logits=return_logits,
                **kwargs,
            )

        return SyncTokGenerationOutput(
            text_token_ids=torch.cat(all_output_token_ids, dim=1) if len(all_output_token_ids) > 0 else None,
            acoustic_features=torch.cat([f if f.ndim == 3 else f.unsqueeze(1) for f in all_acoustic_features], dim=1),
            time_before=torch.cat([f if f.ndim == 2 else f.unsqueeze(1) for f in all_time_before], dim=1),
            llm_time=torch.tensor(llm_time, device=self.device).mean(),
            diffusion_time=torch.tensor(diffusion_time, device=self.device).mean(),
            logits=torch.cat(all_logits, dim=1) if all_logits else None,
            step_logs=step_logs,
        )

    def _decode_wav(self, encoded, time_before):
        # assert time_before.shape[-1] == 1 + encoded.shape[0]
        time_before = time_before[: encoded.shape[0] + 1]
        if time_before.shape[0] == 0:
            return torch.zeros(encoded.shape[0], 0, device=self.device)
        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            encoded_expanded.append(
                torch.zeros(
                    (time_before[pos] - 1).clamp(min=0),
                    encoded.shape[-1],
                    device=self.device,
                    dtype=encoded.dtype,
                )
            )
            encoded_expanded.append(encoded[pos].unsqueeze(0))

        encoded_expanded.append(
            torch.zeros(time_before[-1], encoded.shape[-1], device=self.device, dtype=encoded.dtype)
        )

        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
        return self.decoder.generate(
            encoded_expanded,
            token_masks=(torch.norm(encoded_expanded, dim=-1) != 0).long(),
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        prompt: EncoderOutput,
        text: str | list[str] = "",
        num_transition_steps: int = 5,
        num_extra_steps: int = 0,
        system_prompt: str | None = None,
        user_turn_prompt: str | None = None,
        inference_options: InferenceOptions = InferenceOptions(),
        use_text_in_prompt: bool = False,
        normalize_text: bool = True,
        verbose: bool = False,
    ) -> GenerationOutput:
        if isinstance(text, str):
            text = [text]
        text = [normalize_text_fn(t) if normalize_text else t for t in text]
        input_ids = [
            self.tokenizer.encode(prompt.text[0] + text[i])[prompt.text_tokens_len[i] :] for i in range(len(text))
        ]
        audio_feat_len = (prompt.audio_len / prompt.sample_rate * 50).ceil().long()

        text_tokens = [
            self.tokenizer.encode(prompt.text[0], add_special_tokens=False)
            + self.tokenizer.encode(text[0], add_special_tokens=False)
        ]
        input_ids, input_lengths = self._add_bos_eos(
            torch.tensor(text_tokens, device=self.device),
            torch.tensor([len(token) for token in text_tokens], device=self.device),
        )

        token_positions = prompt.token_positions

        selected_positions_with_ending = torch.where(
            torch.arange(token_positions.shape[1], device=token_positions.device).expand(token_positions.shape[0], -1)
            == input_lengths.reshape(-1, 1) - self.num_eos_tokens - 1,  # without sos and eos
            audio_feat_len.unsqueeze(-1),
            token_positions,
        )
        time_gaps = (
            selected_positions_with_ending
            - torch.nn.functional.pad(selected_positions_with_ending, [1, 0], value=1)[:, :-1]
        ).clamp(min=0, max=self.config.num_time_classes - 1)
        time_gaps = torch.nn.functional.pad(time_gaps, [1, 0], value=0)
        time_len_before = time_gaps[:, :-1]
        time_len_after = time_gaps[:, 1:]

        prompt_acoustic_features = prompt.token_values
        prompt_acoustic_masks = torch.ones(
            prompt_acoustic_features.shape[:2], device=prompt_acoustic_features.device, dtype=torch.long
        )

        if num_extra_steps > 0:
            input_ids = input_ids[:, : -self.num_eos_tokens]

        prefix_text = (
            f"<|start_header_id|>system<|end_header_id|>{system_prompt or ''}<|eot_id|>"
            + (f"<|start_header_id|>user<|end_header_id|>{user_turn_prompt}<|eot_id|>" if user_turn_prompt else "")
            + "<|start_header_id|>assistant<|end_header_id|>"
        )
        prefix_text_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        prefix_len = prefix_text_tokens.shape[1]
        input_ids = torch.cat([input_ids[:, :1], prefix_text_tokens, input_ids[:, 1:]], dim=1)
        input_lengths = input_lengths + len(prefix_text_tokens)
        prompt_acoustic_features = torch.nn.functional.pad(prompt_acoustic_features, (0, 0, prefix_len, 0))
        prompt_acoustic_masks = torch.nn.functional.pad(prompt_acoustic_masks, (prefix_len, 0))
        time_len_before = torch.nn.functional.pad(time_len_before, (prefix_len, 0))
        time_len_after = torch.nn.functional.pad(time_len_after, (prefix_len, 0))

        if num_transition_steps > 0:
            prompt_acoustic_features = prompt_acoustic_features[:, :-num_transition_steps, :]
            prompt_acoustic_masks = prompt_acoustic_masks[:, :-num_transition_steps]
            time_len_before = time_len_before[:, :-num_transition_steps]
            time_len_after = time_len_after[:, :-num_transition_steps]

        outputs: SyncTokGenerationOutput = self._generate(
            input_ids=input_ids,
            text=text,
            input_lengths=input_lengths,
            prompt_acoustic_features=prompt_acoustic_features,
            prompt_acoustic_masks=torch.cat(
                [prompt_acoustic_masks[:, 1:], torch.ones_like(prompt_acoustic_masks[:, :1])], -1
            ),
            prompt_time_len_before=time_len_before,
            prompt_time_len_after=time_len_after,
            num_steps=input_ids.shape[-1] + num_extra_steps,
            inference_options=inference_options,
            use_text_in_prompt=use_text_in_prompt,
        )

        num_prompt_tokens = prompt_acoustic_features.shape[1]
        acoustic_features = outputs.acoustic_features * self.config.acoustic_std + self.config.acoustic_mean

        encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1 :, :]
        time_before = outputs.time_before[..., num_prompt_tokens + num_transition_steps - 1 :]
        wavs = []

        for i in range(encoded.shape[0]):
            try:
                wav = self._decode_wav(encoded[i], time_before=time_before[i]).squeeze(0, 1)
                wav = wav[..., int(24000 * time_before[i][0] / 50) :]  # remove leading silence
                wavs.append(wav)
            except Exception:
                wavs.append(None)

        return GenerationOutput(
            audio=wavs,
            text=text,
            input_text_ids=input_ids,
            input_str=[self.tokenizer.decode(input_ids[i]) for i in range(input_ids.shape[0])],
            output_str=[
                self.tokenizer.decode(outputs.text_token_ids[i]) for i in range(outputs.text_token_ids.shape[0])
            ],
            output_text_ids=outputs.text_token_ids,
            acoustic_features=acoustic_features,
            time_before=time_before,
            prompt_text_tokens=input_ids,
            llm_time=outputs.llm_time,
            diffusion_time=outputs.diffusion_time,
            logits=outputs.logits,
            step_logs=outputs.step_logs,
        )

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    @property
    def sos_id(self):
        return self.tokenizer.bos_token_id

    def _add_bos_eos(self, input_ids: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.nn.functional.pad(input_ids, (0, self.num_eos_tokens), value=self.eos_id)
        input_ids = torch.where(input_ids == -1, self.eos_id, input_ids)
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), value=self.sos_id)
        input_lengths = input_lengths + self.num_eos_tokens + 1
        return input_ids, input_lengths

    @property
    def num_eos_tokens(self):
        return self.config.shift_acoustic

    def compile(self):
        self.model.forward = torch.compile(self.model.forward)
        self.prediction_head.forward = torch.compile(self.prediction_head.forward, mode="reduce-overhead")
        self.prediction_head.forward = torch.compile(self.prediction_head.forward, mode="reduce-overhead")

    def to(self, device: str):
        self.decoder.to(device)
        return super().to(device)
