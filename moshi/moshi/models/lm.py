# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from os.path import splitext, exists
import logging
import numpy as np
import sys
from typing import Optional, Union, List, Tuple, Iterator
import sphn
import torch

from ..utils.sampling import (
    apply_repetition_penalty,
    sample_token,
    sample_top_k_dynamic,
)
from ..utils.compile import CUDAGraphed
from ..modules.streaming import StreamingContainer, StreamingModule, load_streaming_state
from ..modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
)

logger = logging.getLogger(__name__)

AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
MAX_REPETITION_CONTEXT = 256
# Natural PAD/EPAD frames that mark a turn boundary for the repetition
# ring (~1 s at 12.5 Hz). Words inside a turn are separated by only a few
# pad frames; a run this long means the utterance ended. The ring clears
# there so the penalty suppresses within-turn token loops without
# penalizing the next turn's natural opening words.
REPETITION_TURN_BREAK_FRAMES = 12
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)

# Floor for the acoustic sampling temperature tensor. The depformer divides
# logits by it inside a CUDA graph, where a Python zero-check cannot run; at
# this floor the softmax collapses to a one-hot at the argmax, so temperature
# 0 still means greedy decoding instead of NaN probabilities.
MIN_AUDIO_TEMPERATURE = 1e-6
# Ceiling for the first depformer level's sampling temperature. Level 0 is
# Mimi's semantic codebook and decides WHAT the audio says; sampling it as
# hot as the acoustic residuals destabilizes content and reads as crackle,
# while the residual levels tolerate heat for prosody and timbre.
DEFAULT_SEMANTIC_TEMPERATURE_CAP = 0.7
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


def _delay_sequence(delays: List[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: List[int], tensor: torch.Tensor,
                      fill_value: Union[int, float] = float('NaN')) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def create_sinewave(duration: float, sample_rate: int) -> np.ndarray:
    """Return a 440 Hz 'silent' sinewave of the given duration."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize audio to a target LUFS level. Returns mono (T,)."""
    import pyloudnorm as pyln

    # sphn.read returns (C, T). Downmix to mono so the rest of the
    # pipeline (mimi encode of channel 0) gets a faithful mono signal
    # regardless of the upload's channel count.
    if wav.ndim == 2:
        wav = wav.mean(axis=0)

    # pyloudnorm needs at least one full block (default 0.4 s) to
    # measure integrated loudness. Silent or near-silent audio returns
    # -inf, which would scale the signal to inf. In either case, skip
    # the LUFS pass and return the raw mono signal so the upload still
    # produces some kind of voice prompt instead of a hard failure.
    block_samples = int(0.4 * sr)
    if wav.shape[-1] < block_samples:
        return wav
    loudness = pyln.Meter(sr).integrated_loudness(wav)
    if not np.isfinite(loudness):
        return wav
    out = pyln.normalize.loudness(wav, loudness, target_lufs)
    # pyloudnorm.normalize.loudness only WARNS on clipping; it does not
    # rescale. A quiet upload (e.g. -50 LUFS gained to -24 LUFS = +26 dB)
    # can hand mimi.encode samples well above |1.0|, which the codec
    # treats as undefined and surfaces as a degraded voice clone with
    # no error. Peak-normalize down to leave a small headroom margin
    # if the LUFS pass pushed us over.
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def load_audio(
    filepath: str, sample_rate: int, 
):
    """Yields audio samples in intervals of sample_interval_size"""
    sample_pcm, sample_sr = sphn.read(filepath)
    sample_pcm = sphn.resample(
        sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
    )  # shape: (C, T)
    return sample_pcm

def _iterate_audio(sample_pcm, sample_interval_size, max_len=sys.maxsize, pad=True):
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad:
            sample = np.concatenate(
                [
                    sample,
                    np.zeros(
                        (
                            sample.shape[0],
                            sample_interval_size - sample.shape[-1],
                        )
                    ),
                ],
                axis=1,
            )
        cnt += 1
        yield sample[0:1]  # shape: (1, T)


def encode_from_sphn(mimi, samples, max_batch=sys.maxsize):
    """
    Takes an iterator of samples, batches them, encodes them;
    and yields the encoded samples one sample at a time in the same order.
    """
    device = next(mimi.parameters()).device
    current_batch = []
    done_flag = False
    # TO-DO: Fix the batching bug
    max_batch = 1

    while True:
        try:
            sample = next(samples)
            tensor = torch.tensor(sample, dtype=torch.float32, device=device)
            tensor = tensor.unsqueeze(0)  # shape: (1, C, T)                                                                                                      
            current_batch.append(tensor)
        except StopIteration:
            done_flag = True

        if (not done_flag) and len(current_batch) < max_batch:
            continue
        if not current_batch:
            break

        batch = torch.cat(current_batch, dim=0)  # shape: (B, C, T)
        encoded = mimi.encode(batch)  # shape: (B, K, F)
        separated = torch.unbind(encoded, dim=0)  # shape: (K, F)
        reshaped = [x.unsqueeze(0) for x in separated]  # shape: (1, K, F)
        detached = [x.detach().clone() for x in reshaped]

        current_batch = []
        yield from detached  # shape: (1, K, F)

        if done_flag:
            break


class ScaledEmbedding(torch.nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: Optional[int] = None,
        context: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.EmbeddingFactory = EmbeddingFactory
        self.emb = torch.nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = torch.nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = torch.nn.ModuleList(
                [torch.nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = torch.nn.ModuleList(
                [torch.nn.Linear(dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = torch.nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        self.depformer.set_streaming_propagate(False)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token
    
    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        return input_

    def forward_codes(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed_codes(sequence))
    
    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # print("EMBED:", input[0, 0, :10].float().cpu().tolist()) # DEBUG
        transformer_out = self.transformer(input)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T = sequence.shape
        Ka = self.dep_q
        assert (
            K == self.num_codebooks
        ), f"Codebooks for Depformer training should be passed all at once, got {K,}."
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    def forward_train(self, codes: torch.Tensor):
        B, K, T = codes.shape
        # Delaying codes and removing the last time step that will never be an input.
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        # Inserting the empty tokens for the first time step.
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        # LLM Backbone
        transformer_out, text_logits = self.forward_codes(delayed_codes[:, :, :-1])
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens. We will with NaN values invalid positions
        # to ensure they properly handled.
        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.dep_q],
            logits, fill_value=float('NaN'))
        logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


@dataclass
class _LMGenState:
    cache: torch.Tensor
    provided: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_embeddings: CUDAGraphed
    graphed_depth: CUDAGraphed
    recent_text_tokens: torch.Tensor  # [B, MAX_REPETITION_CONTEXT], -1 = empty
    recent_text_offset: torch.Tensor  # [B], advances only on meaningful text
    repetition_pad_streak: torch.Tensor  # [B], natural PAD/EPAD frames in a row
    offset: int = 0

    def reset(self):
        self.offset = 0
        self.recent_text_offset.zero_()
        self.recent_text_tokens.fill_(-1)
        self.repetition_pad_streak.zero_()
        self.provided[:] = False


@torch.no_grad()
def create_loss_report(
    state_cache: torch.Tensor,
    lm_model: LMModel,
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    target: torch.Tensor,
    sampled_text_token: torch.Tensor,
    sampled_audio_tokens: torch.Tensor,
    target_position: int,
) -> dict[str, torch.Tensor]:
    report = {}
    B = state_cache.shape[0]
    # model_tokens is the sampled output from model_logits
    model_tokens = torch.zeros_like(state_cache[:, :, target_position])
    model_tokens[:, 0] = sampled_text_token
    model_tokens[:, 1 : lm_model.dep_q + 1] = sampled_audio_tokens

    report.update(
        {
            "forced_tokens": torch.zeros((B, lm_model.dep_q + 1)),
            "model_tokens": torch.zeros((B, lm_model.dep_q + 1)),
            "ranks_of_forced": torch.zeros((B, lm_model.dep_q + 1)),
            "losses": torch.zeros((B, lm_model.dep_q+1)),
        }
    )
    report["model_tokens"] = model_tokens.clone()
    report["forced_tokens"] = target.clone()

    # Text Channel
    text_logits = text_logits.squeeze(dim=1).squeeze(dim=1)
    target_all = target
    target = target_all[:, 0].squeeze(1).clone()

    text_probs = torch.softmax(text_logits, dim=-1)
    text_ranks = torch.argsort(text_probs, dim=-1, descending=True)
    for b in range(B):
        forced_token = target[b].item()
        try:
            rank = (text_ranks[b] == forced_token).nonzero().item()
        except RuntimeError:
            rank = lm_model.zero_token_id
        report["ranks_of_forced"][b, 0] = rank

    target[target == lm_model.text_initial_token_id] = -100
    text_loss = torch.nn.functional.cross_entropy(
        text_logits,
        target,
        ignore_index=-100,
        )
    report["losses"][:, 0] = text_loss

    # Audio Channels
    for k in range(lm_model.dep_q):
        target = target_all[:, k+1].squeeze(1).clone()
        channel_logits = audio_logits[:, k, :]

        audio_probs = torch.softmax(channel_logits, dim=-1)
        audio_ranks = torch.argsort(audio_probs, dim=-1, descending=True)
        for b in range(B):
            forced_token = target[b].item()
            try:
                rank = (audio_ranks[b] == forced_token).nonzero().item()
            except RuntimeError:
                rank = lm_model.zero_token_id
            report["ranks_of_forced"][b, k + 1] = rank

        target[target == lm_model.initial_token_id] = -100
        audio_loss = torch.nn.functional.cross_entropy(
            channel_logits,
            target,
            ignore_index=-100,
        )
        report["losses"][:, k + 1] = audio_loss
    return report


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        device: str | torch.device,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
        report_loss: bool = False,
        return_logits: bool = False,
        audio_silence_frame_cnt: int = 1,
        text_prompt_tokens: Optional[list[int]] = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 32000,
        frame_rate: int = FRAME_RATE_HZ,
        repetition_penalty: float = 1.0,
        repetition_penalty_context: int = 64,
        padding_bonus: float = 0.0,
        max_turn_text_tokens: int = 0,
        semantic_temperature_cap: float = DEFAULT_SEMANTIC_TEMPERATURE_CAP,
        caption_cfg: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_context = max(0, min(repetition_penalty_context, MAX_REPETITION_CONTEXT))
        self.padding_bonus = padding_bonus
        self.max_turn_text_tokens = max_turn_text_tokens
        self.semantic_temperature_cap = float(semantic_temperature_cap)
        # Min-p truncation for the text head only: keep tokens whose
        # probability is at least min_p_text of the step's maximum. 0
        # disables it and leaves top-k as the sole truncation.
        self.min_p_text: float = 0.0
        # Caption-CFG (context-aware decoding): the streaming state runs two
        # batch rows. Row 0 is the conditional branch and the only row that
        # receives injected context text; row 1 is the unconditional branch
        # and receives PAD in those frames. Text logits are guided as
        # uncond + cfg_gamma * (cond - uncond), sampled once, and the chosen
        # token feeds both rows, so the branches differ only by
        # context-window text and the emitted timeline never carries it.
        # Requires entering streaming with batch size 2.
        self.caption_cfg = bool(caption_cfg)
        # Guidance strength. 1.0 reproduces the conditional branch exactly
        # (context in attention at natural strength); higher values amplify
        # the context delta. Plain Python float mutated live by the server
        # and read once per frame outside the CUDA graphs, like
        # padding_bonus.
        self.cfg_gamma: float = 1.0
        self._non_pad_streak = 0
        self._turn_pad_streak = 0
        self._pad_force_remaining = 0
        self.text_prompt_tokens = text_prompt_tokens
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.voice_prompt = None
        self.zero_text_code = 3
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._frame_size = int(self._sample_rate / self._frame_rate)
        self._zero_frame = torch.zeros(1, 1, self._frame_size, device=self.lm_model.device)
        duration = self._frame_size / self._sample_rate
        sine = create_sinewave(duration, self._sample_rate)
        self._sine_frame = torch.tensor(sine, device=self.lm_model.device).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        self._zero_codes = torch.as_tensor(
            SILENCE_TOKENS,
            dtype=torch.long,
            device=self.lm_model.device,
        ).view(1, 8, 1)
        self._sine_codes = torch.as_tensor(
            SINE_TOKENS,
            dtype=torch.long,
            device=self.lm_model.device,
        ).view(1, 8, 1)
        # Tensor inputs to the CUDA-graphed depformer so live sampling updates
        # change replayed computation without replacing a working graph. The
        # temperature is one value per depformer level: level 0 (the semantic
        # codebook) is capped by semantic_temperature_cap, the acoustic
        # residual levels use the requested temperature directly.
        self._audio_temperature = torch.empty(
            lm_model.dep_q,
            dtype=torch.float32,
            device=self.lm_model.device,
        )
        self._write_audio_temperature(float(temp))
        self._audio_top_k = torch.tensor(
            int(top_k),
            dtype=torch.long,
            device=self.lm_model.device,
        )
        self.check = check
        self.report_loss = report_loss
        if report_loss:
            return_logits = True
        self.return_logits = return_logits
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings
        self.voice_prompt_audio: Optional[torch.Tensor] = None
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None
        # Flattened streaming state loaded from a voice's .safetensors
        # sidecar. Kept as a dict and applied during priming rather than at
        # load time, because callers reset_streaming() between load and
        # priming, which would wipe an immediately-applied state.
        self.voice_prompt_full_state: Optional[dict] = None
        # Fraction of an uploaded clip's prefix replayed during priming, in
        # 0.0..1.0. 1.0 replays the whole clip; lower values replay only the
        # tail (most recent audio), which conditions less strongly; 0.0
        # replays nothing, leaving the model's own voice. Only the raw-audio
        # upload path reads this; the embeddings-replay and blend paths ignore
        # it. Set at connect time before priming, never live.
        self.voice_prompt_strength: float = 1.0
        # Missing: Mimi encoder streaming state is not captured alongside the LM
        # state. When a saved voice prompt is loaded mid-session the LM cache
        # resumes mid-stream but the Mimi encoder restarts at t=0, so the
        # encoder transients drift over the first few minutes of cloned output.
        # Fix would require: bumping the on-disk format to namespace LM and
        # Mimi state (e.g. "lm." / "mimi." prefixes in the safetensors blob),
        # threading the mimi instance into load_voice_prompt_embeddings and
        # set_streaming_state_inplace, and saving mimi.get_streaming_state()
        # at the end of _step_voice_prompt_core before save_streaming_state.

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            False,
            device=lm_model.device,
            dtype=torch.bool
        )

        disable = lm_model.device.type != 'cuda'
        # disable = True # DEBUG
        graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
        graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        recent_text_tokens = torch.full(
            (batch_size, MAX_REPETITION_CONTEXT),
            -1,
            device=lm_model.device,
            dtype=torch.long,
        )
        recent_text_offset = torch.zeros(
            (batch_size,), device=lm_model.device, dtype=torch.long
        )
        repetition_pad_streak = torch.zeros(
            (batch_size,), device=lm_model.device, dtype=torch.long
        )

        return _LMGenState(
            cache,
            provided,
            initial,
            graphed_main,
            graphed_embeddings,
            graphed_depth,
            recent_text_tokens,
            recent_text_offset,
            repetition_pad_streak,
        )
    
    @torch.no_grad()
    def prepare_step_input(self,
                           input_tokens: torch.Tensor=None,
                           moshi_tokens:torch.Tensor=None,
                           text_token:torch.Tensor=None,
                           ):
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        # audio_tokens_per_stream = lm_model.dep_q//2
        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]

        ####
        # Fill Cache with provided tokens at state.offset (target) + delays

        if input_tokens is not None:
            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

            for q_other in range(input_tokens.shape[1]):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q_other
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]
                state.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = moshi_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the moshi stream, got {Ki}."

            for q_moshi in range(moshi_tokens.shape[1]):
                k = 1 + q_moshi
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = moshi_tokens[:, q_moshi]
                state.provided[:, k, write_position : write_position + 1] = True

        if text_token is not None:
            write_position = (state.offset + lm_model.delays[0]) % CT
            state.cache[:, 0, write_position] = text_token
            state.provided[:, 0, write_position] = True

        for k, delay in enumerate(lm_model.delays):
            # Only for the very beginning, we extend the initial token for the acoustic
            # token that are delayed, and thus have no good value to take.
            if state.offset <= delay:
                state.cache[:, k, state.offset % CT] = state.initial[:, k, 0]
                state.provided[:, k, state.offset % CT] = True

        ####
        # Perform inference at state.offset - 1 (model_input); forcing with tokens at state.offset (target) when provided

        if state.offset == 0:
            # We can't report loss or force depth tranformer tokens until we're at step 2
            # And we need to initialize the delay-0 cache where it's not provided for step 2
            state.cache[:, :, 0] = state.initial[:, :, 0] # torch.where(state.provided[:, :, 0], state.cache[:, :, 0], state.initial[:, :, 0])
            state.offset += 1
            return None

        model_input_position = (state.offset-1) % CT
        target_position = state.offset % CT
        input_ = state.cache[:, :, model_input_position : model_input_position + 1]
        target_ = state.cache[:, :, target_position : target_position + 1]
        provided_ = state.provided[:, :, target_position : target_position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        return input_, provided_, target_, model_input_position, target_position

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor=None, moshi_tokens:torch.Tensor=None, text_token:torch.Tensor=None,
             return_embeddings: bool=False) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self._streaming_state
        lm_model = self.lm_model
        prepared_inputs = self.prepare_step_input(
            input_tokens, moshi_tokens, text_token,
        )
        # print("INPUT:", None if input_tokens is None else input_tokens.squeeze().cpu().tolist()) # DEBUG
        # print("MOSHI:", None if moshi_tokens is None else moshi_tokens.squeeze().cpu().tolist()) # DEBUG
        if prepared_inputs is None:
            return (None, None) if self.report_loss or self.return_logits else None
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        embeddings = None
        if return_embeddings:
            embeddings = self.lm_model.embed_codes(input_)
        transformer_out, text_logits = state.graphed_main(input_)
        output = self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
            text_was_forced=text_token is not None,
        )
        if return_embeddings:
            return output, embeddings
        return output
    
    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor):
        state = self._streaming_state
        lm_model = self.lm_model
        if self.caption_cfg and embeddings.shape[0] == 1:
            # Voice sidecars are saved from single-row sessions; both CFG
            # rows prime identically from the same embedding.
            embeddings = embeddings.expand(2, -1, -1)
        needed_input_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        _dummy_audio_token = lm_model._get_initial_token()
        while True:
            prepared_inputs = self.prepare_step_input(
                input_tokens=_dummy_audio_token[:, 1:1+needed_input_tokens], moshi_tokens=_dummy_audio_token[:, 1+needed_input_tokens:], text_token=self.zero_text_code,
            )
            if prepared_inputs is not None:
                break
        _, provided_, target_, model_input_position, target_position = prepared_inputs
        transformer_out, text_logits = state.graphed_embeddings(embeddings)
        return self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
            text_was_forced=True,
        )

    @torch.no_grad()
    def process_transformer_output(
        self,
        transformer_out,
        text_logits,
        provided_,
        target_,
        model_input_position,
        target_position,
        *,
        text_was_forced: bool,
    ):
        state = self._streaming_state
        lm_model = self.lm_model

        # Shape of text_logits should be [B, K_text=1, T=1, Card_text].
        # text_logits may alias a CUDA-graph captured output buffer. Apply the
        # repetition penalty first because it already returns a clone; only
        # make a separate copy when padding bias is the sole mutation.
        text_logits_f = text_logits.float()
        cfg_active = self.caption_cfg and text_logits_f.shape[0] == 2
        if cfg_active:
            # Guided text logits: the unconditional row plus cfg_gamma times
            # the context-conditioned delta. The single guided row is
            # penalized and sampled once below; the clone at gamma 1.0 keeps
            # the padding-bonus write off the graph output buffer.
            gamma = float(self.cfg_gamma)
            cond = text_logits_f[0:1]
            if gamma == 1.0:
                text_logits_f = cond.clone()
            else:
                uncond = text_logits_f[1:2]
                text_logits_f = uncond + gamma * (cond - uncond)
        if self.repetition_penalty > 1.0 and self.repetition_penalty_context > 0:
            text_logits_f = self._apply_text_repetition_penalty(text_logits_f)
        elif (
            self.padding_bonus != 0.0
            and text_logits_f.data_ptr() == text_logits.data_ptr()
        ):
            text_logits_f = text_logits_f.clone()
        # Bias the text padding token up to encourage the model to yield its
        # turn. Moshi emits text_padding_token when it has nothing to say; a
        # positive bonus shortens rambling. 0 = off, 2-4 typical.
        if self.padding_bonus != 0.0:
            text_logits_f[..., lm_model.text_padding_token_id] += self.padding_bonus
        sampled_text_token = sample_token(
            text_logits_f,
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
            min_p=self.min_p_text,
        )
        assert sampled_text_token.dim() == 3, sampled_text_token.shape
        assert sampled_text_token.shape[2] == 1
        assert sampled_text_token.shape[1] == 1, "Only one text stream supported."
        sampled_text_token = sampled_text_token[:, 0, 0]  # shape is [B]
        if cfg_active:
            # One guided sample drives both rows; forced context windows
            # still win per row through the provided/target mask below.
            sampled_text_token = sampled_text_token.expand(2)

        next_text_token = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text_token)

        # Hard cap on turn length. If the model emits N text tokens within a
        # turn, force pad for ~1 s of text frames (12.5 Hz) so the audio
        # decoder produces real silence and the turn actually yields. Brief
        # PAD gaps between phrases do not reset the count; only a sustained
        # natural PAD/EPAD run marks a new turn.
        # The sampled-token accounting happens after the depformer launch to
        # avoid synchronizing CUDA between the main and depth graphs. A newly
        # reached cap arms padding for the following frame, preserving the
        # same maximum number of emitted text tokens.
        pad_id = lm_model.text_padding_token_id
        next_text_token, turn_pad_forced = self._consume_forced_pad(
            next_text_token,
            pad_id,
            text_was_forced=text_was_forced,
        )

        # Update repetition penalty ring buffer with the chosen text token.
        # Exclude forced (externally-injected) tokens, mirroring the
        # max-turn streak guard above. Without this, injected caption/persona
        # words enter the ring buffer and the model is later penalized for
        # naturally referencing the very scene it was just fed.
        if (
            self.repetition_penalty > 1.0
            and self.repetition_penalty_context > 0
            and not text_was_forced
        ):
            self._update_repetition_ring(
                next_text_token, pad_was_forced=turn_pad_forced
            )

        if self.return_logits:
            sampled_audio_tokens, audio_logits = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0], self._audio_temperature, self._audio_top_k) # [B, K_audio, Card_audio]
        else:
            sampled_audio_tokens = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0], self._audio_temperature, self._audio_top_k)

        state.provided[:, :, model_input_position] = False
        ####
        # Fill cache with generated tokens at state.offset (where not provided)

        state.cache[:, 0, target_position] = torch.where(
            ~state.provided[:, 0, target_position],
            next_text_token,
            state.cache[:, 0, target_position],
        )
        state.cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(
            ~state.provided[:, 1 : lm_model.dep_q + 1, target_position],
            sampled_audio_tokens,
            state.cache[:, 1 : lm_model.dep_q + 1, target_position],
        )
        if cfg_active:
            # Both rows must hear the same assistant audio. Row 0's depformer
            # (whose transformer state carries the context) is canonical; its
            # codes overwrite row 1's slot so the two KV timelines differ
            # only by context-window text tokens.
            state.cache[1:2, 1 : lm_model.dep_q + 1, target_position] = (
                state.cache[0:1, 1 : lm_model.dep_q + 1, target_position]
            )

        self._update_turn_cap(
            next_text_token,
            pad_id,
            text_was_forced=text_was_forced,
            turn_pad_forced=turn_pad_forced,
        )

        ####
        # Calculate loss of model logits (based on state.offset - 1) compared to target (state.offset)

        report = {}
        if self.report_loss:
            report = create_loss_report(
                state_cache=state.cache,
                lm_model=lm_model,
                text_logits=text_logits,
                audio_logits=audio_logits,
                target=target_,
                sampled_text_token=sampled_text_token,
                sampled_audio_tokens=sampled_audio_tokens,
                target_position=target_position,
            )

        ####
        # Collect outputs for state.offset - max_delay

        if state.offset <= self.max_delay:
            state.offset += 1
            if self.report_loss:
                return None, report
            if self.return_logits:
                return None, None
            else:
                return None
        
        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)

        state.offset += 1
        if self.report_loss:
            return out, report
        elif self.return_logits and not self.report_loss:
            return out, (text_logits.clone(), audio_logits.clone())
        else:
            return out

    def load_voice_prompt(self, voice_prompt: str):
        self.voice_prompt = voice_prompt
        # sphn.read returns (C, T). normalize_audio downmixes to mono and
        # returns (T,). Re-add the channel dim so the encoder gets (1, T).
        raw_audio = load_audio(voice_prompt, self._sample_rate)
        raw_audio = normalize_audio(raw_audio, self._sample_rate, -24.0)
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]
        self.voice_prompt_audio = raw_audio
        self.voice_prompt_cache = None
        self.voice_prompt_embeddings = None
        self.voice_prompt_full_state = None

    def load_voice_prompt_embeddings(self, path: str):
        # First try to load full streaming state if available
        base_path = splitext(path)[0]
        state_path = base_path + ".safetensors"
        meta_path = base_path + ".json"
        
        if exists(state_path) and exists(meta_path):
            # Single-row sidecars restore into a two-row caption-CFG stream
            # as well: the tensor restore copies via Tensor.copy_, which
            # broadcasts (1, ...) into (2, ...), priming both rows
            # identically.
            logger.info("loading full streaming state from %s", state_path)
            full_state = load_streaming_state(state_path, meta_path, device=self.lm_model.device)
            self._migrate_legacy_full_state(full_state)
            # Stash the dict; _step_voice_prompt_core applies it during
            # priming, after the callers' reset_streaming().
            self.voice_prompt_full_state = full_state
            # Mark that we have loaded the full state so _step_voice_prompt_core can skip replay
            self.voice_prompt_embeddings = [] # Non-None but empty to signal "loaded"
            self.voice_prompt_audio = None
            # The full state carries the token cache, so no separate copy is
            # needed. The legacy .pt path (below) restores the cache itself.
            self.voice_prompt_cache = None
            self.voice_prompt = path
            return

        # Fallback to legacy .pt loading (replay required)
        logger.info("loading legacy voice prompt embeddings from %s", path)
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = data["embeddings"].to(self.lm_model.device)
        self.voice_prompt_cache = data["cache"].to(self.lm_model.device)
        self.voice_prompt_full_state = None
        self.voice_prompt = path

    def _migrate_legacy_full_state(self, state: dict) -> None:
        """Upgrade a pre-tensor-ring voice sidecar layout in place.

        Sidecars written before the turn-scoped repetition ring store
        recent_text_offset as a plain int and carry no repetition_pad_streak
        key. Restoring that layout unmigrated replaces the live offset
        tensor with an int and then fails on the missing key, corrupting
        the process-global streaming state for every later session.
        """
        device = self.lm_model.device
        for key in list(state.keys()):
            prefix, _, leaf = key.rpartition(".")
            if leaf != "recent_text_offset":
                continue
            value = state[key]
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                state[key] = torch.full(
                    (1,), value, dtype=torch.long, device=device
                )
            sibling = f"{prefix}.repetition_pad_streak"
            if sibling not in state:
                state[sibling] = torch.zeros(
                    (1,), dtype=torch.long, device=device
                )

    def _load_voice_prompt_embedding_sequence(self, path: str) -> torch.Tensor:
        """Load a voice's stacked per-frame embeddings from a legacy .pt file.

        Returns the (T, ...) embedding sequence on the model device, matching
        the device placement of the legacy load path in
        load_voice_prompt_embeddings. Blending operates on this per-frame
        replay representation, not the full streaming-state (.safetensors)
        form, which is a wholesale state overwrite and cannot be mixed.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data["embeddings"].to(self.lm_model.device)

    def load_voice_prompt_blend(self, path_a: str, path_b: str, mix: float):
        """Condition on a frame-aligned mix of two voices' per-frame embeddings.

        `mix` is the secondary share in 0..1: 0.0 is all primary, 1.0 is all
        secondary. The two sequences are aligned to the shorter length and
        combined elementwise as (1 - mix) * a + mix * b, then replayed through
        the LM by the existing voice-prompt replay path. The final cache is
        left unset so the replay rebuilds it for the blended trajectory.
        """
        seq_a = self._load_voice_prompt_embedding_sequence(path_a)
        seq_b = self._load_voice_prompt_embedding_sequence(path_b)
        n = min(seq_a.shape[0], seq_b.shape[0])
        blended = (1.0 - mix) * seq_a[:n] + mix * seq_b[:n]
        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = blended
        self.voice_prompt_cache = None
        self.voice_prompt_full_state = None
        self.voice_prompt = f"{path_a}+{path_b}@{mix:.2f}"

    def _encode_zero_frame(self) -> torch.Tensor:
        return self._zero_codes

    def _encode_sine_frame(self) -> torch.Tensor:
        return self._sine_codes

    def _update_repetition_ring(
        self, next_text_token: torch.Tensor, *, pad_was_forced: bool = False
    ) -> None:
        """Track recent meaningful text for the repetition penalty.

        Skips PAD (3) and EPAD (0) so they never crowd the context; filtering
        0 also keeps the duplicate-index scatter in apply_repetition_penalty
        unambiguous (the sentinel for empty slots). The ring is turn-scoped:
        a sustained run of natural PAD/EPAD frames marks a turn boundary and
        clears it, so the penalty kills within-turn token loops without
        penalizing the next turn's natural opening words. Max-turn-cap forced
        PADs are not the model yielding — counting them would wipe the ring
        right after every cap trip, forgiving the exact repetition that
        tripped it — so they freeze the streak instead of advancing it. All
        tensor ops so no per-step CUDA sync.
        """
        state = self._streaming_state
        ctx = self.repetition_penalty_context
        keep_mask = (next_text_token != 0) & (next_text_token != 3)
        # Also enforce the boundary here for direct callers and for penalty-
        # disabled frames. The sampling path performs the same clear earlier
        # so the first next-turn token is not penalized one frame too long.
        self._clear_repetition_boundary()
        if not pad_was_forced:
            state.repetition_pad_streak.add_(1).clamp_(
                max=REPETITION_TURN_BREAK_FRAMES
            )
        state.repetition_pad_streak.masked_fill_(keep_mask, 0)
        slots = torch.remainder(state.recent_text_offset, ctx).unsqueeze(1)
        existing = state.recent_text_tokens.gather(1, slots)
        values = torch.where(
            keep_mask.unsqueeze(1), next_text_token.unsqueeze(1), existing
        )
        state.recent_text_tokens.scatter_(1, slots, values)
        state.recent_text_offset.add_(keep_mask.to(dtype=torch.long))

    def _clear_repetition_boundary(self) -> None:
        state = self._streaming_state
        boundary = state.repetition_pad_streak >= REPETITION_TURN_BREAK_FRAMES
        state.recent_text_tokens.masked_fill_(boundary.unsqueeze(1), -1)
        state.recent_text_offset.masked_fill_(boundary, 0)

    def _apply_text_repetition_penalty(
        self, text_logits: torch.Tensor
    ) -> torch.Tensor:
        # Clear before applying the penalty: clearing only after sampling lets
        # the previous turn bias the first token of the next one.
        self._clear_repetition_boundary()
        ctx = self.repetition_penalty_context
        # Under caption-CFG the guided logits collapse to a single row while
        # the ring keeps one row per branch; the rows hold identical history
        # (both branches consume the same sampled token), so the leading
        # slice is the shared context.
        recent = self._streaming_state.recent_text_tokens[
            : text_logits.shape[0], :ctx
        ]
        return apply_repetition_penalty(
            text_logits,
            recent,
            self.repetition_penalty,
        )

    def _consume_forced_pad(
        self,
        next_text_token: torch.Tensor,
        pad_id: int,
        *,
        text_was_forced: bool,
    ) -> tuple[torch.Tensor, bool]:
        # Interrupts use the same force window as the automatic turn cap and
        # must work even when max_turn_text_tokens is disabled.
        if text_was_forced or self._pad_force_remaining <= 0:
            return next_text_token, False
        self._pad_force_remaining -= 1
        return torch.full_like(next_text_token, pad_id), True

    def _update_turn_cap(
        self,
        next_text_token: torch.Tensor,
        pad_id: int,
        *,
        text_was_forced: bool,
        turn_pad_forced: bool,
    ) -> None:
        """Count text across brief pauses, resetting at a real turn break."""
        if text_was_forced:
            return
        if turn_pad_forced:
            self._non_pad_streak = 0
            self._turn_pad_streak = 0
            return
        if self.max_turn_text_tokens <= 0:
            self._non_pad_streak = 0
            self._turn_pad_streak = 0
            return
        token = int(next_text_token[0].item())
        if token in (0, pad_id):
            self._turn_pad_streak += 1
            if self._turn_pad_streak >= REPETITION_TURN_BREAK_FRAMES:
                self._non_pad_streak = 0
            return
        self._turn_pad_streak = 0
        self._non_pad_streak += 1
        if self._non_pad_streak >= self.max_turn_text_tokens:
            self._pad_force_remaining = REPETITION_TURN_BREAK_FRAMES
            self._non_pad_streak = 0

    def reset_repetition_state(self) -> None:
        """Clear turn-scoped repetition history without reallocating tensors."""
        state = self._streaming_state
        if state is None:
            return
        state.recent_text_tokens.fill_(-1)
        state.recent_text_offset.zero_()
        state.repetition_pad_streak.zero_()

    def _write_audio_temperature(self, temperature: float) -> None:
        """Fill the per-level temperature vector for one acoustic setting.

        The semantic level (0) never exceeds semantic_temperature_cap; when
        the requested temperature sits at or below the cap, every level
        matches it exactly.
        """
        acoustic = max(float(temperature), MIN_AUDIO_TEMPERATURE)
        semantic = max(
            min(acoustic, self.semantic_temperature_cap),
            MIN_AUDIO_TEMPERATURE,
        )
        self._audio_temperature.fill_(acoustic)
        self._audio_temperature[0] = semantic

    def set_audio_sampling(self, temperature: float, top_k: int) -> bool:
        """Apply graph-safe acoustic sampling controls in place.

        Both values are copied into CUDA-graph input tensors. The depformer
        uses a fixed-cardinality top-k and masks ranks against the tensor k,
        and the per-level temperature vector keeps its shape, so neither
        update requires graph recapture.

        Returns whether top-k changed (for applied-config telemetry).
        """
        temperature = float(temperature)
        top_k = int(top_k)
        top_k_changed = top_k != self.top_k
        self.temp = temperature
        self._write_audio_temperature(temperature)
        self.top_k = top_k
        self._audio_top_k.fill_(top_k)
        return top_k_changed

    def _strength_sliced_voice_prompt_audio(self):
        """Tail slice of the uploaded clip selected by voice_prompt_strength.

        The clip primes the cache frame by frame; replaying fewer frames
        conditions less strongly. Keeping the tail makes the most recent
        audio the last thing the cache sees, so the kept slice dominates.
        Strength 1.0 keeps the whole clip (current behavior), 0.0 keeps
        nothing (the model's own voice). Returns the slice as a (C, T')
        array on the same axis layout the encoder iterator expects.
        """
        audio = self.voice_prompt_audio
        strength = max(0.0, min(1.0, self.voice_prompt_strength))
        if strength >= 1.0:
            return audio
        total_samples = audio.shape[-1]
        total_frames = -(-total_samples // self._frame_size)  # ceil
        keep_frames = round(total_frames * strength)
        if keep_frames <= 0:
            return audio[:, :0]
        keep_samples = min(total_samples, keep_frames * self._frame_size)
        return audio[:, -keep_samples:]

    def _encode_voice_prompt_frames(self, mimi):
        return encode_from_sphn(
            mimi,
            _iterate_audio(
                self._strength_sliced_voice_prompt_audio(),
                sample_interval_size=self._frame_size,
                pad=True,
            ),
            max_batch=1,
        )

    def _step_voice_prompt_frame(self,
                                 voice_prompt_frame_tokens: torch.Tensor,
                                 saved_embeddings: Optional[list[torch.Tensor]]=None,
                                 ):
        # Always use zero_text_code during voice prompt
        out = self.step(
            moshi_tokens=voice_prompt_frame_tokens,
            text_token=self.zero_text_code,
            input_tokens=self._encode_sine_frame(),
            return_embeddings=self.save_voice_prompt_embeddings,
        )
        if out is not None and self.save_voice_prompt_embeddings:
            _, embeddings = out
            saved_embeddings.append(embeddings)

    def _step_voice_prompt_core(self, mimi) -> Iterator[None]:
        """Shared core for stepping through the voice prompt.

        This generator yields at each *checkpoint* where the async wrapper may want to
        consult `is_alive`. The core itself is intentionally unaware of connection state.
        """
        if self.voice_prompt_embeddings is not None:
            if self.voice_prompt_full_state is not None:
                # set_streaming_state_inplace pops entries from the dict it
                # is given; pass a fresh shallow copy so the next priming
                # can apply the same state again.
                self.set_streaming_state_inplace(dict(self.voice_prompt_full_state))
                return

            # Replay stored voice prompt embeddings
            for next_embed in self.voice_prompt_embeddings:
                yield
                self.step_embeddings(next_embed)

            # A blended prompt has no stored final cache: the cache it would
            # restore belongs to neither source voice, so replaying the
            # blended sequence is what builds the correct cache. Only the
            # single-voice .pt path carries a saved final cache to copy in.
            if self.voice_prompt_cache is not None:
                state = self._streaming_state
                state.cache.copy_(self.voice_prompt_cache)
            return

        elif self.voice_prompt_audio is not None:
            saved_embeddings = []
            for voice_prompt_frame_tokens in self._encode_voice_prompt_frames(mimi):
                yield
                self._step_voice_prompt_frame(
                    voice_prompt_frame_tokens,
                    saved_embeddings
                )
            # One last checkpoint before any optional save (nice-to-have for async disconnect)
            yield

            if self.save_voice_prompt_embeddings:
                # Save full streaming state (tensors + metadata) to bypass replay next time.
                # We use .pt extension for compatibility with existing logic but store full state.
                base_path = splitext(self.voice_prompt)[0]
                state_path = base_path + ".safetensors"
                meta_path = base_path + ".json"
                
                # Also save the legacy .pt format for backward compatibility if needed, 
                # or just use it as a marker.
                torch.save(
                    {
                        "embeddings": torch.stack(saved_embeddings, dim=0).detach().cpu(),
                        "cache": self._streaming_state.cache,
                        "full_state_available": True
                    },
                    base_path + ".pt",
                )
                self.save_streaming_state(state_path, meta_path)
        logger.info("done loading voice prompt")

    def _step_voice_prompt(self, mimi):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_voice_prompt_core(mimi):
            pass

    async def _step_voice_prompt_async(self, mimi, is_alive: Optional[Callable]=None):
        for _ in self._step_voice_prompt_core(mimi):
            if is_alive is not None and not await is_alive():
                break

    def _step_audio_silence_core(self) -> Iterator[None]:
        # For slots of silence (default 0.5s) after voice/text prompts
        # (agent text, user audio, agent audio) : (PADs, silence, sine)
        for _ in range(self.audio_silence_frame_cnt):
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )
        logger.info("done loading audio silence")

    def _step_audio_silence(self):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_audio_silence_core():
            pass

    async def _step_audio_silence_async(self, is_alive: Optional[Callable]=None):
        for _ in self._step_audio_silence_core():
            if is_alive is not None and not await is_alive():
                break

    def _step_text_prompt_core(self) -> Iterator[None]:
        # text_prompt_tokens defaults to None; treat that as no prompt.
        for text_prompt_token in self.text_prompt_tokens or []:
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=text_prompt_token,
                input_tokens=self._encode_sine_frame(),
            )
        logger.info("done loading text prompt")


    def _step_text_prompt(self):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_text_prompt_core():
            pass

    async def _step_text_prompt_async(self, is_alive: Optional[Callable]=None):
        for _ in self._step_text_prompt_core():
            if is_alive is not None and not await is_alive():
                break

    async def step_system_prompts_async(self, mimi, is_alive: Optional[Callable]=None):
        await self._step_voice_prompt_async(mimi, is_alive)
        await self._step_audio_silence_async(is_alive)
        await self._step_text_prompt_async(is_alive)
        await self._step_audio_silence_async(is_alive)

    def step_system_prompts(self, mimi):
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_provided: torch.Tensor,
        audio_temperature: torch.Tensor,
        audio_top_k: torch.Tensor,
    ) -> torch.Tensor:
        (B,) = text_token.shape
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        depformer_logits: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.return_logits:
                    assert logits.shape == (B, 1, 1, lm_model.card), logits.shape
                    ret_logits = logits.squeeze(dim=1).squeeze(dim=1)
                    assert ret_logits.shape == (B, lm_model.card), ret_logits.shape
                    depformer_logits.append(ret_logits.float())
                if self.use_sampling:
                    probs = torch.softmax(
                        logits.float() / audio_temperature[cb_index], dim=-1
                    )
                    next_token = sample_top_k_dynamic(
                        probs, audio_top_k
                    )[..., 0]
                else:
                    next_token = torch.argmax(logits, dim=-1)
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                prev_token = torch.where(
                    audio_provided[:, cb_index],
                    audio_tokens[:, cb_index],
                    next_token,
                )
                depformer_tokens.append(next_token)

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        tokens = torch.stack(depformer_tokens, dim=1)
        assert tokens.shape == (B, lm_model.dep_q), tokens.shape
        if self.return_logits:
            all_logits = torch.stack(depformer_logits, dim=1)
            assert all_logits.shape == (B, lm_model.dep_q, lm_model.card), all_logits.shape
            return tokens, all_logits
        else:
            return tokens
