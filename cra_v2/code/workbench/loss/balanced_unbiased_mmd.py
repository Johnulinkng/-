"""Equal-row unbiased multi-kernel MMD for channel-count transfer.

The v1 channel loss flattened every sensor, producing 3B source rows versus B
target rows in 3-to-1 transfer.  This module refuses unequal row counts and
provides a deterministic, balanced physical-slot sampler for both domains.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


class BalancedPhysicalSlotCycle:
    """Stateful one-row-per-sample physical-slot cycle.

    The supplied generator is consulted exactly once to choose the initial
    offset.  Advancing a continuous cycle rather than redrawing an offset for
    every minibatch guarantees that cumulative exposure counts differ by at
    most one for every prefix of draws.  ``physical_slots`` keeps a singleton
    target such as ``[2]`` identifiable as physical slot 2 rather than local
    tensor column 0.
    """

    def __init__(
        self,
        physical_slots: tuple[int, ...] | list[int],
        generator: torch.Generator,
        *,
        device: torch.device | str,
    ) -> None:
        slots = tuple(int(value) for value in physical_slots)
        if not slots or len(set(slots)) != len(slots) or any(value < 0 for value in slots):
            raise ValueError("physical_slots must be distinct nonnegative integers")
        if not isinstance(generator, torch.Generator):
            raise TypeError("an explicit torch.Generator is required")
        self.device = torch.device(device)
        if generator.device.type != self.device.type:
            raise ValueError("generator and features must use the same device type")
        self.physical_slots = slots
        self.cursor = (
            0
            if len(slots) == 1
            else int(torch.randint(0, len(slots), (1,), generator=generator, device=self.device))
        )
        self.initial_cursor = self.cursor
        self.rows_drawn = 0

    def sample(self, sensor_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(sensor_features, torch.Tensor)
            or not sensor_features.is_floating_point()
            or sensor_features.ndim != 3
            or min(sensor_features.shape) < 1
            or not torch.isfinite(sensor_features).all()
        ):
            raise ValueError("sensor_features must be finite [batch,channels,width]")
        if sensor_features.device.type != self.device.type:
            raise ValueError("features and cycle must use the same device type")
        batch, channels, _ = sensor_features.shape
        if channels != len(self.physical_slots):
            raise ValueError("feature columns differ from declared physical slots")
        local = (torch.arange(batch, device=sensor_features.device) + self.cursor) % channels
        rows = torch.arange(batch, device=sensor_features.device)
        physical_map = torch.tensor(self.physical_slots, dtype=torch.long, device=sensor_features.device)
        physical = physical_map[local]
        self.cursor = int((self.cursor + batch) % channels)
        self.rows_drawn += int(batch)
        return sensor_features[rows, local], physical

    def state_dict(self) -> dict:
        return {
            "physical_slots": list(self.physical_slots),
            "initial_cursor": self.initial_cursor,
            "cursor": self.cursor,
            "rows_drawn": self.rows_drawn,
        }


def _positive(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _features(source: torch.Tensor, target: torch.Tensor) -> int:
    for name, value in (("source", source), ("target", target)):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1:
            raise ValueError(f"{name} must have shape [batch>=2,features>=1]")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")
    if source.shape != target.shape:
        raise ValueError("equal-row MMD requires identical source/target shapes")
    if source.device != target.device or source.dtype != target.dtype:
        raise ValueError("source and target device/dtype must match")
    return int(source.shape[0])


def balanced_physical_slot_sample(
    sensor_features: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one physical slot per row with counts differing by at most one.

    This stateless helper is retained for one-minibatch diagnostics.  Training
    must use :class:`BalancedPhysicalSlotCycle` so balance also holds across
    minibatch boundaries.
    """
    if (
        not isinstance(sensor_features, torch.Tensor)
        or not sensor_features.is_floating_point()
        or sensor_features.ndim != 3
        or min(sensor_features.shape) < 1
        or not torch.isfinite(sensor_features).all()
    ):
        raise ValueError("sensor_features must be finite [batch,channels,width]")
    if not isinstance(generator, torch.Generator):
        raise TypeError("an explicit torch.Generator is required")
    if generator.device.type != sensor_features.device.type:
        raise ValueError("generator and features must use the same device type")
    batch, channels, _ = sensor_features.shape
    if channels == 1:
        slots = torch.zeros(batch, dtype=torch.long, device=sensor_features.device)
    else:
        offset = torch.randint(
            0, channels, (1,), generator=generator, device=sensor_features.device
        )
        slots = (torch.arange(batch, device=sensor_features.device) + offset) % channels
    rows = torch.arange(batch, device=sensor_features.device)
    return sensor_features[rows, slots], slots


def unbiased_multi_kernel_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidths: tuple[Real, ...] | list[Real],
) -> torch.Tensor:
    """Signed U-statistic for a five-kernel RBF fixed before adaptation.

    No value-dependent clamp, absolute value, squaring, or bandwidth estimate is
    permitted here. A negative finite result is a valid finite-sample estimate.
    """
    batch = _features(source, target)
    if not isinstance(bandwidths, (tuple, list)) or len(bandwidths) != 5:
        raise ValueError("exactly five frozen bandwidths are required")
    frozen = [_positive(value, f"bandwidth[{index}]") for index, value in enumerate(bandwidths)]
    total = torch.cat((source, target), dim=0)
    distance = (total[:, None, :] - total[None, :, :]).square().sum(dim=2)
    if not torch.isfinite(distance).all():
        raise FloatingPointError("pairwise distance is non-finite")
    kernels = sum(torch.exp(-distance / source.new_tensor(value)) for value in frozen)
    if not torch.isfinite(kernels).all():
        raise FloatingPointError("kernel matrix is non-finite")
    xx = kernels[:batch, :batch]
    yy = kernels[batch:, batch:]
    xy = kernels[:batch, batch:]
    diagonal = torch.eye(batch, dtype=torch.bool, device=source.device)
    loss = xx.masked_select(~diagonal).sum() / (batch * (batch - 1))
    loss = loss + yy.masked_select(~diagonal).sum() / (batch * (batch - 1))
    loss = loss - 2.0 * xy.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("unbiased MMD is non-finite")
    return loss
