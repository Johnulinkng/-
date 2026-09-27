"""Isolated, fixed-dimensional frequency-grid ablation of SpectralSharedEncoder.

Only the spectral pooling grid changes. Both grids produce 128 log-power bins;
all trainable layers and their initialization are inherited unchanged.
"""
import hashlib

import torch
from torch.utils.data import DataLoader

from .spectral_shared import (SPECTRAL_BINS, WINDOW_LENGTH, SpectralSharedEncoder,
                              log_power_spectrum)


class _SensorView(torch.utils.data.Dataset):
    """Calibration view only: retain one stored sensor, never inspect its label."""
    def __init__(self, base, slot):
        self.base, self.slot = base, slot
        if hasattr(base, 'indices'):
            self.indices = base.indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        waveform, _ = self.base[index]
        return waveform[self.slot:self.slot + 1], -1


GRIDS = {'full12k_pool8': (1024, 8), 'low3k_pool2': (256, 2)}


def log_power_grid(x, spectral_grid, window=None):
    """Sum disjoint positive-frequency powers, unit-normalize, then log."""
    if spectral_grid not in GRIDS:
        raise ValueError('Unknown spectral grid')
    # Exact old path is intentional: a full-grid pilot must be numerically
    # equivalent to the existing 128-dimensional representation.
    if spectral_grid == 'full12k_pool8':
        return log_power_spectrum(x, window)
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3:
        raise ValueError('Spectrum requires a floating-point [N,C,2048] tensor')
    if min(x.shape[:2]) < 1 or x.shape[-1] != WINDOW_LENGTH:
        raise ValueError('Spectrum requires positive batch/channels and exactly 2048 samples')
    if not torch.isfinite(x).all():
        raise ValueError('Spectrum inputs must be finite')
    if window is None:
        window = torch.hann_window(WINDOW_LENGTH, periodic=False, dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=-1, keepdim=True)
    positive = torch.fft.rfft(centered * window, dim=-1)[..., 1:257]
    power = positive.abs().square().reshape(*x.shape[:2], SPECTRAL_BINS, 2).sum(dim=-1)
    relative = power / power.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    result = (relative + 1e-8).log()
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite normalized log spectrum')
    return result


@torch.no_grad()
def training_grid_statistics(dataset, batch_size, spectral_grid):
    """Use each training waveform once; target fault labels are never read."""
    if spectral_grid not in GRIDS or len(dataset) < 1 or batch_size < 1:
        raise ValueError('Unknown grid, empty training data or invalid batch size')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(0))
    window = torch.hann_window(WINDOW_LENGTH, periodic=False)
    count, rows, mean, m2, channels = 0, 0, None, None, None
    digest = hashlib.sha256()
    for waveforms, _ignored_labels in loader:
        values = log_power_grid(waveforms.detach().to(device='cpu', dtype=torch.float32),
                                spectral_grid, window)
        if channels is not None and values.shape[1] != channels:
            raise ValueError('One calibration dataset must use one channel selection')
        channels = values.shape[1]
        rows += values.shape[0]
        digest.update(values.contiguous().numpy().tobytes())
        values = values.flatten(0, 1).double()
        batch_count = len(values)
        batch_mean = values.mean(dim=0)
        batch_m2 = (values - batch_mean).square().sum(dim=0)
        if mean is None:
            mean, m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta.square() * (count * batch_count / (count + batch_count))
            mean = mean + delta * (batch_count / (count + batch_count))
        count += batch_count
    std = (m2 / count).clamp_min(0).sqrt()
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    indices = getattr(dataset, 'indices', None)
    evidence = {'rows': rows, 'channels': channels, 'sensor_windows': count,
                'feature_bytes_sha256': digest.hexdigest(), 'device': 'cpu',
                'ordered_original_indices_sha256': None if indices is None else
                hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                'labels_used': False, 'drop_last': False, 'spectral_grid': spectral_grid}
    return mean.float(), std.float(), evidence


class SpectralGridEncoder(SpectralSharedEncoder):
    """Old shared encoder with one explicit choice of 128-bin FFT grid."""

    def __init__(self, input_channels, num_classes, spectral_grid='full12k_pool8',
                 zscore_adaptation=True, encoder_kind='mlp', domain_adversarial=False,
                 source_statistics_scope='pooled', encoder_width=64, fusion='mean'):
        if spectral_grid not in GRIDS:
            raise ValueError('Unknown spectral grid')
        if source_statistics_scope not in ('pooled', 'per_sensor'):
            raise ValueError('Unknown source statistics scope')
        super().__init__(input_channels, num_classes, zscore_adaptation=zscore_adaptation,
                         encoder_kind=encoder_kind, domain_adversarial=domain_adversarial,
                         encoder_width=encoder_width, fusion=fusion)
        self.spectral_grid = spectral_grid
        self.source_statistics_scope = source_statistics_scope
        if source_statistics_scope == 'per_sensor':
            self.register_buffer('source_sensor_center', torch.zeros(input_channels, SPECTRAL_BINS))
            self.register_buffer('source_sensor_scale', torch.ones(input_channels, SPECTRAL_BINS))

    @torch.no_grad()
    def calibrate(self, source_train, target_train=None, batch_size=32):
        # Compute before the existing calibration freezes its flag. This path
        # changes source preprocessing only; target statistics stay identical.
        sensor_stats = None
        if self.source_statistics_scope == 'per_sensor':
            if self.source_calibrated.item():
                raise ValueError('Statistics are frozen after calibration')
            sensor_stats = [training_grid_statistics(_SensorView(source_train, slot),
                            batch_size, self.spectral_grid) for slot in range(self.input_channels)]
        if self.spectral_grid == 'full12k_pool8':
            evidence = super().calibrate(source_train, target_train, batch_size)
            evidence['spectral_grid'] = self.spectral_grid
            return self._record_sensor_statistics(evidence, sensor_stats)
        if self.source_calibrated.item():
            raise ValueError('Statistics are frozen after calibration; construct a new model to refit')
        if self.target_statistics_adaptation != (target_train is not None):
            raise ValueError('Only explicit target-statistics adaptation receives target training data')
        source_mean, source_std, source_evidence = training_grid_statistics(
            source_train, batch_size, self.spectral_grid)
        if source_evidence['channels'] != self.input_channels:
            raise ValueError('Calibration source channels differ from the model configuration')
        target_mean, target_std, target_evidence = source_mean, source_std, None
        if target_train is not None:
            target_mean, target_std, target_evidence = training_grid_statistics(
                target_train, batch_size, self.spectral_grid)
        self.source_center.copy_(source_mean)
        self.source_scale.copy_(source_std)
        self.target_center.copy_(target_mean)
        self.target_scale.copy_(target_std)
        self.source_calibrated.fill_(True)
        self.target_calibrated.fill_(target_train is not None)
        self.calibration_evidence = {
            'source_train': source_evidence, 'target_train': target_evidence,
            'source_transform': '(log_power - source_train_mean) / source_train_std',
            'target_transform': ('(log_power - target_train_mean) / target_train_std' if self.zscore_adaptation
                                 else '(log_power - source_train_mean) / source_train_std'),
            'target_labels_used': False, 'validation_or_final_statistics_used': False,
            'frozen_after_calibration': True, 'spectral_grid': self.spectral_grid}
        return self._record_sensor_statistics(self.calibration_evidence, sensor_stats)

    @torch.no_grad()
    def _record_sensor_statistics(self, evidence, sensor_stats):
        if sensor_stats is not None:
            self.source_sensor_center.copy_(torch.stack([item[0] for item in sensor_stats]))
            self.source_sensor_scale.copy_(torch.stack([item[1] for item in sensor_stats]))
            evidence['source_per_sensor'] = [item[2] for item in sensor_stats]
            evidence['source_statistics_scope'] = 'per_sensor'
            evidence['source_transform'] = '(log_power[channel] - source_train_mean[channel]) / source_train_std[channel]'
        self.calibration_evidence = evidence
        return evidence

    def standardized_spectrum(self, x, domain):
        if domain == 'source' and self.source_statistics_scope == 'per_sensor':
            if not self.source_calibrated.item():
                raise RuntimeError('Train-only statistics must be calibrated before encoding')
            if x.ndim != 3 or x.shape[1] != self.input_channels:
                raise ValueError('Per-sensor source input must retain recorded channel order and width')
            return ((log_power_grid(x, self.spectral_grid, self.hann) - self.source_sensor_center)
                    / self.source_sensor_scale)
        if self.spectral_grid == 'full12k_pool8':
            return super().standardized_spectrum(x, domain)
        if domain not in {'source', 'target'}:
            raise ValueError('Unknown spectral domain')
        if not self.source_calibrated.item() or (self.target_statistics_adaptation and
                                                  not self.target_calibrated.item()):
            raise RuntimeError('Train-only statistics must be calibrated before encoding')
        target_adapted = domain == 'target' and self.target_statistics_adaptation
        center = self.target_center if target_adapted else self.source_center
        scale = self.target_scale if domain == 'target' and self.zscore_adaptation else self.source_scale
        return (log_power_grid(x, self.spectral_grid, self.hann) - center) / scale

    def architecture_metadata(self):
        details = super().architecture_metadata()
        positive_bins, pool_width = GRIDS[self.spectral_grid]
        details['spectral_grid'] = self.spectral_grid
        details['source_statistics_scope'] = self.source_statistics_scope
        if self.source_statistics_scope == 'per_sensor':
            details['normalization'] = 'fixed source-train per-frequency population statistics separately for each source sensor; target unchanged'
        details['positive_frequency_bins_retained'] = positive_bins
        details['power_pool_width'] = pool_width
        details['power_pool'] = (f'sum {pool_width} adjacent powers among first {positive_bins} '
                                 'positive-frequency bins, then unit-sum power and log(p+1e-8)')
        return details
