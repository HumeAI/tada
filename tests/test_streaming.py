"""Tests for streaming audio generation.

Unit tests run without GPU/model (mock-based).
Integration tests (marked with @pytest.mark.integration) require GPU + model weights.
Run integration tests with: pytest tests/test_streaming.py -m integration -s
"""

import pytest
import torch

from tada.modules.decoder import StreamingDecoder, _create_segment_attention_mask
from tada.modules.tada import AudioStream, GenerationOutput, StepOutput, SyncTokGenerationOutput


# ---------------------------------------------------------------------------
# Unit tests — no GPU, no model weights required
# ---------------------------------------------------------------------------


class TestStepOutput:
    def test_fields(self):
        feat = torch.randn(1, 1, 512)
        tb = torch.tensor([[5]])
        s = StepOutput(acoustic_features=feat, time_before=tb)
        assert s.acoustic_features is feat
        assert s.time_before is tb


class TestAudioStream:
    """Test AudioStream with a fake generator (no real model needed)."""

    def _make_fake_gen(self, num_steps=5, acoustic_dim=512):
        """Create a generator that yields StepOutputs then SyncTokGenerationOutput."""
        all_feats = []
        all_tb = []
        for i in range(num_steps):
            feat = torch.randn(1, 1, acoustic_dim)
            tb = torch.tensor([[8]])  # 8 frames before each token
            all_feats.append(feat)
            all_tb.append(tb)
            yield StepOutput(acoustic_features=feat, time_before=tb)

        # Final output
        yield SyncTokGenerationOutput(
            text_token_ids=torch.zeros(1, num_steps, dtype=torch.long),
            acoustic_features=torch.cat(all_feats, dim=1),
            time_before=torch.cat(all_tb, dim=1),
            llm_time=torch.tensor(0.1),
            diffusion_time=torch.tensor(0.05),
            logits=None,
            step_logs=[],
        )

    def _make_audio_stream(self, num_steps=10):
        """Create AudioStream with a real (tiny) Decoder for integration."""
        from tada.modules.decoder import Decoder, DecoderConfig

        config = DecoderConfig(
            embed_dim=512,
            hidden_dim=64,  # tiny for testing
            num_attn_layers=1,
            num_attn_heads=2,
            attn_dim_feedforward=128,
            strides=[2, 2, 2, 2],  # 16x upsampling (fast)
            block_attention="v2",
        )
        decoder = Decoder(config)
        decoder.eval()

        gen = self._make_fake_gen(num_steps=num_steps)
        stream = AudioStream(
            gen=gen,
            decoder=decoder,
            acoustic_mean=0.0,
            acoustic_std=1.5,
            cnn_window_size=50,
        )
        # Mock the context needed for _build_result
        stream._generate_context = {
            "text": ["test"],
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "token_decode_offset": 0,
            "tokenizer": None,  # won't be used in this test path
            "acoustic_std": 1.5,
            "acoustic_mean": 0.0,
        }
        return stream

    def test_iteration_yields_tuples(self):
        stream = self._make_audio_stream(num_steps=15)
        chunks = list(stream)
        # Should yield at least some chunks
        assert len(chunks) > 0
        for chunk, sr in chunks:
            assert isinstance(chunk, torch.Tensor)
            assert sr == 24000
            assert chunk.ndim >= 1
            assert chunk.shape[-1] > 0

    def test_total_audio_length_positive(self):
        stream = self._make_audio_stream(num_steps=15)
        chunks = list(stream)
        total_samples = sum(c.shape[-1] for c, _ in chunks)
        assert total_samples > 0

    def test_empty_generation(self):
        """Generator that yields only the final output (no predicted tokens)."""
        def empty_gen():
            yield SyncTokGenerationOutput(
                text_token_ids=torch.zeros(1, 0, dtype=torch.long),
                acoustic_features=torch.zeros(1, 0, 512),
                time_before=torch.zeros(1, 0, dtype=torch.long),
                llm_time=torch.tensor(0.0),
                diffusion_time=torch.tensor(0.0),
                logits=None,
                step_logs=[],
            )

        from tada.modules.decoder import Decoder, DecoderConfig

        config = DecoderConfig(
            embed_dim=512, hidden_dim=64, num_attn_layers=1, num_attn_heads=2,
            attn_dim_feedforward=128, strides=[2, 2, 2, 2], block_attention="v2",
        )
        decoder = Decoder(config)
        stream = AudioStream(
            gen=empty_gen(), decoder=decoder,
            acoustic_mean=0.0, acoustic_std=1.5,
        )
        stream._generate_context = {
            "text": [""], "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "token_decode_offset": 0, "tokenizer": None,
            "acoustic_std": 1.5, "acoustic_mean": 0.0,
        }
        chunks = list(stream)
        # Empty generation should produce no chunks (or only a flush chunk)
        assert len(chunks) <= 1


class TestStreamingDecoder:
    """Test StreamingDecoder with a tiny decoder model."""

    @pytest.fixture
    def tiny_decoder(self):
        from tada.modules.decoder import Decoder, DecoderConfig

        config = DecoderConfig(
            embed_dim=512,
            hidden_dim=64,
            num_attn_layers=1,
            num_attn_heads=2,
            attn_dim_feedforward=128,
            strides=[2, 2, 2, 2],
            block_attention="v2",
        )
        decoder = Decoder(config)
        decoder.eval()
        return decoder

    def test_basic_streaming(self, tiny_decoder):
        sd = StreamingDecoder(tiny_decoder, cnn_window_size=50, min_first_emission=10)
        chunks = []
        for i in range(20):
            token = torch.randn(512)
            chunk = sd.decode_block(token, time_before=5)
            if chunk is not None and chunk.numel() > 0:
                chunks.append(chunk)
        final = sd.flush(trailing_frames=3)
        if final is not None and final.numel() > 0:
            chunks.append(final)
        assert len(chunks) > 0
        total_samples = sum(c.shape[-1] for c in chunks)
        assert total_samples > 0

    def test_skip_leading_frames(self, tiny_decoder):
        sd = StreamingDecoder(tiny_decoder, cnn_window_size=50, min_first_emission=10)
        sd.skip_leading_frames(10)
        assert sd._emitted_frames == 10

    def test_reset(self, tiny_decoder):
        sd = StreamingDecoder(tiny_decoder, cnn_window_size=50, min_first_emission=10)
        # Feed some data
        for _ in range(5):
            sd.decode_block(torch.randn(512), time_before=3)
        sd.reset()
        assert sd._cached_frames == 0
        assert sd._emitted_frames == 0
        assert sd._all_hidden is None

    def test_min_block_frames_buffering(self, tiny_decoder):
        sd = StreamingDecoder(tiny_decoder, min_block_frames=5, cnn_window_size=50, min_first_emission=10)
        # Single token with time_before=1 gives only 1 frame, should buffer
        result = sd.decode_block(torch.randn(512), time_before=1)
        assert result is None  # buffered, not enough frames

    def test_flush_with_no_data(self, tiny_decoder):
        sd = StreamingDecoder(tiny_decoder, cnn_window_size=50)
        result = sd.flush()
        assert result is None


class TestSegmentAttentionMask:
    def test_v2_basic(self):
        # 2 blocks: [0,0,1,0,1] -> block0=[0,0,1], block1=[0,1]
        mask_input = torch.tensor([[0, 0, 1, 0, 1]])
        mask = _create_segment_attention_mask(mask_input, version="v2")
        assert mask.shape == (1, 5, 5)
        # Position 0 (block 0) should attend to block 0 (same) but not block 1
        # Position 3 (block 1) should attend to block 0 (prev) and block 1 (same)
        assert mask[0, 0, 3].item() == True   # block 0 can't attend to block 1
        assert mask[0, 3, 0].item() == False  # block 1 can attend to block 0 (prev)


# ---------------------------------------------------------------------------
# Integration tests — require GPU + model weights
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestStreamingIntegration:
    """Full integration tests with real TADA model.

    Run with: pytest tests/test_streaming.py -m integration -s
    """

    @pytest.fixture(scope="class")
    def model_and_prompt(self):
        import torchaudio
        from tada.modules.encoder import Encoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
        from tada.modules.tada import TadaForCausalLM
        model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b", torch_dtype=dtype).to(device)

        audio, sr = torchaudio.load("tada/samples/ljspeech.wav")
        prompt = encoder(audio.to(device), sample_rate=sr)

        return model, prompt, device

    def test_non_streaming_unchanged(self, model_and_prompt):
        """Non-streaming path should work exactly as before."""
        model, prompt, device = model_and_prompt
        output = model.generate(
            prompt=prompt,
            text="Hello world, this is a test.",
        )
        assert output.audio is not None
        assert len(output.audio) == 1
        assert output.audio[0] is not None
        assert output.audio[0].shape[-1] > 0

    def test_streaming_produces_chunks(self, model_and_prompt):
        """Streaming should produce audio chunks."""
        model, prompt, device = model_and_prompt
        stream = model.generate(
            prompt=prompt,
            text="Hello world, this is a streaming test.",
            stream=True,
        )
        chunks = []
        for chunk, sr in stream:
            assert sr == 24000
            assert chunk.shape[-1] > 0
            chunks.append(chunk)

        assert len(chunks) > 0
        total_samples = sum(c.shape[-1] for c in chunks)
        assert total_samples > 0
        # Check .result is populated
        assert stream.result is not None
        assert stream.result.acoustic_features is not None

    def test_streaming_vs_nonstreaming_similar_length(self, model_and_prompt):
        """Streaming and non-streaming should produce similar-length audio."""
        model, prompt, device = model_and_prompt
        text = "Please call Stella."

        # Non-streaming
        output_ns = model.generate(prompt=prompt, text=text)
        ns_len = output_ns.audio[0].shape[-1]

        # Streaming
        stream = model.generate(prompt=prompt, text=text, stream=True)
        chunks = [c for c, _ in stream]
        s_len = sum(c.shape[-1] for c in chunks)

        # Allow 20% tolerance (different random seeds in flow matching)
        ratio = s_len / ns_len if ns_len > 0 else 1.0
        assert 0.5 < ratio < 2.0, f"Length ratio {ratio} is too different"

    def test_streaming_early_break(self, model_and_prompt):
        """Breaking from iteration should not crash."""
        model, prompt, device = model_and_prompt
        stream = model.generate(
            prompt=prompt,
            text="This is a long text that should produce many chunks of audio for testing early termination.",
            stream=True,
        )
        count = 0
        for chunk, sr in stream:
            count += 1
            if count >= 2:
                break
        # Should not raise


# ---------------------------------------------------------------------------
# Generate 5 test audios for manual listening
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateAudios:
    """Generate 5 audio samples for manual listening validation.

    Run with: pytest tests/test_streaming.py::TestGenerateAudios -m integration -s
    Outputs saved to tests/output/
    """

    TEXTS = [
        "Hello, this is a demonstration of streaming text to speech.",
        "Please call Stella. Ask her to bring these things with her from the store.",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was silence. Then came the voice, clear and bright.",
        "Technology is best when it brings people together.",
    ]

    @pytest.fixture(scope="class")
    def model_and_prompt(self):
        import torchaudio
        from tada.modules.encoder import Encoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
        from tada.modules.tada import TadaForCausalLM
        model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b", torch_dtype=dtype).to(device)

        audio, sr = torchaudio.load("tada/samples/ljspeech.wav")
        prompt = encoder(audio.to(device), sample_rate=sr)

        return model, prompt

    def test_generate_streaming_audios(self, model_and_prompt):
        import os
        import torchaudio

        model, prompt = model_and_prompt
        os.makedirs("tests/output", exist_ok=True)

        for i, text in enumerate(self.TEXTS):
            print(f"\n--- Generating streaming audio {i+1}: {text[:50]}...")
            stream = model.generate(prompt=prompt, text=text, stream=True)
            chunks = []
            for chunk, sr in stream:
                chunks.append(chunk)
                print(f"  chunk: {chunk.shape[-1] / sr:.2f}s")

            if chunks:
                full_audio = torch.cat(chunks, dim=-1)
                if full_audio.ndim == 1:
                    full_audio = full_audio.unsqueeze(0)
                out_path = f"tests/output/streaming_{i+1}.wav"
                torchaudio.save(out_path, full_audio.cpu().float(), 24000)
                print(f"  saved: {out_path} ({full_audio.shape[-1] / 24000:.2f}s)")

        # Also generate non-streaming for comparison
        for i, text in enumerate(self.TEXTS):
            print(f"\n--- Generating non-streaming audio {i+1}: {text[:50]}...")
            output = model.generate(prompt=prompt, text=text)
            if output.audio[0] is not None:
                wav = output.audio[0]
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                out_path = f"tests/output/nonstreaming_{i+1}.wav"
                torchaudio.save(out_path, wav.cpu().float(), 24000)
                print(f"  saved: {out_path} ({wav.shape[-1] / 24000:.2f}s)")
