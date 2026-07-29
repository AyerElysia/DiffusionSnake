from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from .contracts import SliceSequenceMeta
from .memory_bank import SliceMemoryBank


class RelativeSliceDistanceEncoding(nn.Module):
    """Encode spatial slice distance independently from Flow time."""

    def __init__(self, dim: int, distance_scale: float = 4.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.distance_scale = float(distance_scale)
        half = max(self.dim // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(max(half - 1, 1))
        self.register_buffer(
            "inv_frequency",
            torch.pow(10000.0, -exponent),
            persistent=False,
        )

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        scaled = distance.reshape(-1, 1) / self.distance_scale
        phase = scaled * self.inv_frequency.reshape(1, -1).to(distance)
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if encoded.size(-1) < self.dim:
            encoded = torch.nn.functional.pad(encoded, (0, self.dim - encoded.size(-1)))
        return encoded[:, :self.dim]


class MemoryCrossAttention(nn.Module):
    """Block-level contour-to-memory attention with per-slice K/V caching."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError("MemFlowDiT dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.query_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.key_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.output_proj = nn.Linear(self.dim, self.dim, bias=True)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._cached_key: Optional[torch.Tensor] = None
        self._cached_value: Optional[torch.Tensor] = None
        self._cached_valid: Optional[torch.Tensor] = None
        self._contour_indices: Optional[torch.Tensor] = None
        self.last_delta = 0.0

    def set_slice_memory(
        self,
        key_tokens: Optional[torch.Tensor],
        value_tokens: Optional[torch.Tensor],
        valid_tokens: Optional[torch.Tensor],
    ) -> None:
        if key_tokens is None or value_tokens is None or valid_tokens is None:
            self.clear_slice_memory()
            return
        batch_size, token_count, _ = key_tokens.shape
        key = self.key_proj(self.memory_norm(key_tokens))
        value = self.value_proj(self.memory_norm(value_tokens))
        self._cached_key = key.view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        self._cached_value = value.view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        self._cached_valid = valid_tokens.to(dtype=torch.bool)
        self._contour_indices = None
        self.last_delta = 0.0

    def set_contour_indices(self, contour_indices: Optional[torch.Tensor]) -> None:
        self._contour_indices = contour_indices

    def clear_slice_memory(self) -> None:
        self._cached_key = None
        self._cached_value = None
        self._cached_valid = None
        self._contour_indices = None
        self.last_delta = 0.0

    def forward(self, contour_tokens: torch.Tensor) -> torch.Tensor:
        if self._cached_key is None:
            self.last_delta = 0.0
            return contour_tokens
        contour_count, point_count, _ = contour_tokens.shape
        image_count = self._cached_key.size(0)
        indices = self._contour_indices
        if indices is None:
            if image_count == contour_count:
                indices = torch.arange(contour_count, device=contour_tokens.device)
            elif image_count == 1:
                indices = torch.zeros(
                    contour_count,
                    dtype=torch.long,
                    device=contour_tokens.device,
                )
            else:
                raise ValueError("MemFlowDiT requires contour-to-slice indices")
        indices = indices.to(device=contour_tokens.device, dtype=torch.long)
        key = self._cached_key[indices]
        value = self._cached_value[indices]
        valid = self._cached_valid[indices]
        query = self.query_proj(self.query_norm(contour_tokens)).view(
            contour_count,
            point_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = valid[:, None, None, :]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(
            contour_count,
            point_count,
            self.dim,
        )
        residual = self.output_proj(attended)
        has_memory = valid.any(dim=1).to(dtype=residual.dtype).view(-1, 1, 1)
        residual = residual * has_memory
        self.last_delta = float(residual.detach().abs().mean().item())
        return contour_tokens + residual


class MemFlowDiTController(nn.Module):
    """Own Memory adapters while preserving inherited DiT block instances."""

    def __init__(
        self,
        dit_blocks: Sequence[nn.Module],
        memory_dim: int,
        state_dim: int,
        num_heads: int,
        distance_scale: float,
    ) -> None:
        super().__init__()
        if int(memory_dim) != int(state_dim):
            raise ValueError("prototype requires memory_dim == DiT state_dim")
        self.memory_dim = int(memory_dim)
        self.distance_encoding = RelativeSliceDistanceEncoding(
            self.memory_dim,
            distance_scale=distance_scale,
        )
        self.adapters = nn.ModuleList([
            MemoryCrossAttention(state_dim, num_heads) for _ in dit_blocks
        ])
        self._hook_handles = []
        for block, adapter in zip(dit_blocks, self.adapters):
            self._hook_handles.append(
                block.register_forward_hook(self._make_hook(adapter))
            )
        self.active_state_count = 0

    @staticmethod
    def _make_hook(adapter: MemoryCrossAttention):
        def hook(_module, _inputs, output):
            return adapter(output)

        return hook

    def set_slice_memory(
        self,
        banks: Sequence[SliceMemoryBank],
        metas: Sequence[SliceSequenceMeta],
    ) -> None:
        if len(banks) != len(metas):
            raise ValueError("parallel MemFlowDiT inputs must have identical lengths")
        packed: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        token_counts: List[int] = []
        reference = None
        active_states = 0
        for bank, meta in zip(banks, metas):
            meta.validate()
            if bank.volume_id != meta.volume_id:
                raise ValueError("memory bank volume does not match slice metadata")
            keys = []
            values = []
            for state in bank.states():
                if (
                    state.volume_id != meta.volume_id
                    or state.position_unit != meta.position_unit
                ):
                    raise ValueError("Slice Memory metadata mismatch")
                key = state.key.flatten(2).transpose(1, 2).squeeze(0)
                value = state.value.flatten(2).transpose(1, 2).squeeze(0)
                if (
                    key.size(-1) != self.memory_dim
                    or value.size(-1) != self.memory_dim
                ):
                    raise ValueError("Slice Memory token dimension mismatch")
                delta_z = key.new_full(
                    (key.size(0),),
                    meta.slice_position - state.slice_position,
                )
                position = self.distance_encoding(delta_z)
                keys.append(key + position)
                values.append(value + position)
                reference = key
                active_states += 1
            if keys:
                packed.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
                token_counts.append(packed[-1][0].size(0))
            else:
                packed.append((None, None))
                token_counts.append(0)
        self.active_state_count = active_states
        if reference is None:
            for adapter in self.adapters:
                adapter.clear_slice_memory()
            return
        max_tokens = max(token_counts)
        batch_size = len(banks)
        key_batch = reference.new_zeros((batch_size, max_tokens, self.memory_dim))
        value_batch = reference.new_zeros((batch_size, max_tokens, self.memory_dim))
        valid_batch = torch.zeros(
            (batch_size, max_tokens),
            dtype=torch.bool,
            device=reference.device,
        )
        for index, ((key, value), count) in enumerate(zip(packed, token_counts)):
            if count == 0:
                continue
            key_batch[index, :count] = key
            value_batch[index, :count] = value
            valid_batch[index, :count] = True
        for adapter in self.adapters:
            adapter.set_slice_memory(key_batch, value_batch, valid_batch)

    def set_contour_indices(self, contour_indices: Optional[torch.Tensor]) -> None:
        for adapter in self.adapters:
            adapter.set_contour_indices(contour_indices)

    def mean_read_delta(self) -> float:
        if not self.adapters:
            return 0.0
        return sum(adapter.last_delta for adapter in self.adapters) / float(
            len(self.adapters)
        )


class MemFlowDiTDenoiser(nn.Module):
    """Instance-local wrapper that injects Memory inside inherited Flow DiT."""

    def __init__(
        self,
        base_denoiser: nn.Module,
        controller: MemFlowDiTController,
    ) -> None:
        super().__init__()
        self.base_denoiser = base_denoiser
        object.__setattr__(self, "_memflow_controller", controller)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base_denoiser")
            return getattr(base, name)

    def forward(self, *args, **kwargs):
        self._memflow_controller.set_contour_indices(kwargs.get("py_ind"))
        return self.base_denoiser(*args, **kwargs)


def install_memflow_dit(
    contour_adapter: nn.Module,
    memory_dim: int,
    state_dim: int,
    num_heads: int,
    distance_scale: float,
) -> MemFlowDiTController:
    """Install MemFlowDiT only on this adapter's Flow evolution instance."""
    network = contour_adapter.slice_loss_wrapper.net
    evolution = network.gcn
    denoiser = evolution.denoiser
    if isinstance(denoiser, MemFlowDiTDenoiser):
        raise RuntimeError("MemFlowDiT is already installed on this model instance")
    blocks = list(denoiser.dit_layers)
    if not blocks:
        raise RuntimeError("MemFlowDiT requires inherited DiT blocks")
    controller = MemFlowDiTController(
        dit_blocks=blocks,
        memory_dim=memory_dim,
        state_dim=state_dim,
        num_heads=num_heads,
        distance_scale=distance_scale,
    )
    evolution.denoiser = MemFlowDiTDenoiser(denoiser, controller)
    return controller
