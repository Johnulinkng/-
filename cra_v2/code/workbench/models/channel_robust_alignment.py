"""Channel-count robust spectral transfer model.

The model is intentionally small and auditable.  It uses one encoder for every
sensor, a bounded residual attention pool, and the same prediction path for a
one-sensor or a three-sensor target.  Channel masking and domain alignment are
training policies implemented by the caller; inference always uses every
observed target sensor.

This module does not claim that the component combination is novel or superior.
Those claims require the frozen comparisons performed by the experiment runner.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

from .spectral_grid_pilot import SpectralGridEncoder


def _finite_unit_interval(value: Real, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not lower_ok or result > 1:
        raise ValueError(f"{name} must be in {'[0,1]' if allow_zero else '(0,1]'}")
    return result


class ChannelRobustAlignmentNet(SpectralGridEncoder):
    """Shared spectral encoder with conservative learnable sensor fusion.

    ``fusion_residual_strength`` bounds how far the learned pooling can move
    from the arithmetic mean.  A zero-initialized score vector makes the exact
    initial function equal to mean pooling while still permitting gradients to
    reach the score vector whenever the residual strength is positive.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        spectral_grid: str = "low3k_pool2",
        encoder_width: int = 64,
        fusion: str = "mean",
        fusion_residual_strength: float = 0.5,
    ) -> None:
        if fusion not in {"mean", "bounded_attention"}:
            raise ValueError("fusion must be mean or bounded_attention")
        strength = _finite_unit_interval(
            fusion_residual_strength, "fusion_residual_strength", allow_zero=True
        )
        super().__init__(
            input_channels,
            num_classes,
            spectral_grid=spectral_grid,
            zscore_adaptation=True,
            encoder_kind="mlp",
            domain_adversarial=False,
            encoder_width=encoder_width,
            fusion="mean",
        )
        self.fusion_mode = fusion
        self.fusion_residual_strength = strength
        if fusion == "bounded_attention":
            # No RNG draw: matched mean/attention arms retain identical encoder
            # and classifier initialization.
            self.fusion_score = nn.Parameter(torch.zeros(self.encoder_width))

    @staticmethod
    def _validate_sensor_mask(sensors: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if sensors.ndim != 3 or min(sensors.shape[:2]) < 1 or not torch.isfinite(sensors).all():
            raise ValueError("sensor features must be finite [batch,channels,width]")
        if mask is None:
            return torch.ones(sensors.shape[:2], dtype=torch.bool, device=sensors.device)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != sensors.shape[:2]
            or mask.device != sensors.device
            or not torch.all((mask == 0) | (mask == 1)).item()
            or not torch.all(mask.sum(dim=1) > 0).item()
        ):
            raise ValueError("mask must retain at least one sensor in every row")
        return mask.bool()

    def fusion_weights(
        self, sensors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = self._validate_sensor_mask(sensors, mask)
        uniform = mask.to(sensors.dtype) / mask.sum(dim=1, keepdim=True)
        if self.fusion_mode == "mean" or sensors.shape[1] == 1:
            return uniform
        scores = (sensors @ self.fusion_score) / math.sqrt(self.encoder_width)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite fusion scores")
        learned = scores.masked_fill(~mask, -float("inf")).softmax(dim=1)
        weights = uniform + self.fusion_residual_strength * (learned - uniform)
        if not torch.isfinite(weights).all() or not torch.allclose(
            weights.sum(dim=1), torch.ones(len(weights), device=weights.device), atol=1e-6, rtol=0
        ):
            raise FloatingPointError("invalid bounded fusion weights")
        return weights

    def fuse_sensors(
        self, sensors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        valid = self._validate_sensor_mask(sensors, mask)
        if mask is None:
            base = sensors.mean(dim=1)
        else:
            base = (sensors * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        if self.fusion_mode == "mean" or sensors.shape[1] == 1:
            return base
        uniform = valid.to(sensors.dtype) / valid.sum(dim=1, keepdim=True)
        weights = self.fusion_weights(sensors, valid)
        # This residual form makes a zero score vector bitwise equal to the
        # historical arithmetic-mean path rather than merely algebraically so.
        return base + ((weights - uniform).unsqueeze(-1) * sensors).sum(dim=1)

    def sensor_features(self, waveforms: torch.Tensor, domain: str) -> torch.Tensor:
        return self.encoder(self.standardized_spectrum(waveforms, domain))

    def encode_features(
        self,
        waveforms: torch.Tensor,
        domain: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sensors = self.sensor_features(waveforms, domain)
        fused = self.fuse_sensors(sensors, mask)
        return sensors, fused, self.classifier(fused)

    def encode_source(self, waveforms: torch.Tensor):
        _, features, logits = self.encode_features(waveforms, "source")
        return features, logits

    def encode_target(self, waveforms: torch.Tensor):
        _, features, logits = self.encode_features(waveforms, "target")
        return features, logits

    def predict_source_branch(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.encode_source(waveforms)[1]

    def predict_target(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.encode_target(waveforms)[1]

    def attention_balance_loss(
        self, sensors: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Squared distance from available-sensor uniform weights.

        The regularizer prevents a source sensor from being silently discarded;
        it is exactly zero for mean fusion and any single-sensor target.
        """
        actual = self.fusion_weights(sensors, mask)
        valid = self._validate_sensor_mask(sensors, mask)
        uniform = valid.to(sensors.dtype) / valid.sum(dim=1, keepdim=True)
        return (actual - uniform).square().sum(dim=1).mean()

    def architecture_metadata(self) -> dict:
        details = super().architecture_metadata()
        details.update(
            identity="channel_robust_alignment_v1",
            fusion_mode=self.fusion_mode,
            source_channel_subsets="training_policy_controlled_by_runner",
            target_inference="all observed target sensors; no channel imputation",
        )
        if self.fusion_mode == "bounded_attention":
            details.update(
                fusion="uniform-anchored bounded content attention after shared sensor encoding",
                fusion_parameter_count=self.fusion_score.numel(),
                fusion_initialization="zero score; exact arithmetic mean",
                fusion_residual_strength=self.fusion_residual_strength,
                single_sensor_fusion="identity",
            )
        else:
            details.update(
                fusion="arithmetic mean after shared sensor encoding",
                fusion_parameter_count=0,
                fusion_initialization="not applicable",
                fusion_residual_strength=0.0,
                single_sensor_fusion="identity",
            )
        details["parameter_count"] = sum(p.numel() for p in self.parameters())
        return details
