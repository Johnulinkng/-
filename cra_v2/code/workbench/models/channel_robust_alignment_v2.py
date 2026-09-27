"""CRA-v2A model with separable source-anchor and target calibration.

The trainable backbone is the audited v1 shared encoder and bounded fusion.
The only model-level change is a fail-closed calibration lifecycle: a common
source anchor is fitted once, then each target arm receives train-only target
statistics without changing the source statistics or trainable weights.
"""

from __future__ import annotations

import torch

from .channel_robust_alignment import ChannelRobustAlignmentNet
from .spectral_grid_pilot import log_power_grid, training_grid_statistics


class ChannelRobustAlignmentV2(ChannelRobustAlignmentNet):
    def calibrate_source_anchor(self, source_train, *, batch_size: int) -> dict:
        if self.source_calibrated.item() or self.target_calibrated.item():
            raise ValueError("source anchor calibration is single-use")
        center, scale, evidence = training_grid_statistics(
            source_train, batch_size, self.spectral_grid
        )
        if evidence["channels"] != self.input_channels:
            raise ValueError("source anchor must retain all declared source channels")
        self.source_center.copy_(center)
        self.source_scale.copy_(scale)
        self.source_calibrated.fill_(True)
        self.target_calibrated.fill_(False)
        self.calibration_evidence = {
            "source_train": evidence,
            "target_train": None,
            "source_transform": "(log_power - source_train_mean) / source_train_std",
            "target_transform": None,
            "target_labels_used": False,
            "validation_or_final_statistics_used": False,
            "source_anchor_frozen": True,
            "target_pending": True,
            "spectral_grid": self.spectral_grid,
        }
        return self.calibration_evidence

    def calibrate_target_train(self, target_train, *, batch_size: int) -> dict:
        if not self.source_calibrated.item():
            raise ValueError("source anchor calibration must be loaded first")
        if self.target_calibrated.item():
            raise ValueError("target calibration is single-use per arm")
        center, scale, evidence = training_grid_statistics(
            target_train, batch_size, self.spectral_grid
        )
        self.target_center.copy_(center)
        self.target_scale.copy_(scale)
        self.target_calibrated.fill_(True)
        source_evidence = None
        if isinstance(self.calibration_evidence, dict):
            source_evidence = self.calibration_evidence.get("source_train")
        self.calibration_evidence = {
            "source_train": source_evidence,
            "target_train": evidence,
            "source_transform": "(log_power - source_train_mean) / source_train_std",
            "target_transform": "(log_power - target_train_mean) / target_train_std",
            "target_labels_used": False,
            "validation_or_final_statistics_used": False,
            "source_anchor_frozen": True,
            "target_pending": False,
            "spectral_grid": self.spectral_grid,
        }
        return self.calibration_evidence

    def standardized_spectrum(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        if domain not in {"source", "target"}:
            raise ValueError("unknown spectral domain")
        if not self.source_calibrated.item():
            raise RuntimeError("source anchor statistics are not calibrated")
        if domain == "target" and not self.target_calibrated.item():
            raise RuntimeError("target-train statistics are not calibrated")
        values = log_power_grid(x, self.spectral_grid, self.hann)
        if domain == "source":
            return (values - self.source_center) / self.source_scale
        return (values - self.target_center) / self.target_scale

    def architecture_metadata(self) -> dict:
        details = super().architecture_metadata()
        details.update(
            identity="channel_robust_alignment_v2a",
            calibration_lifecycle=(
                "one source-only anchor calibration; one target-train-only calibration per arm"
            ),
            adaptation_selection="fixed 20-epoch endpoint; no target-label selection",
        )
        return details

