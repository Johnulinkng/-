"""Shared sensor spectrum encoder with frozen, train-only domain centering.

This is a candidate under evaluation, not a claim of a new generic algorithm.
Source and target use the same trainable parameters. Target mean calibration
is an explicit UDA operation; it is never called Source Only.
"""
import hashlib
import math
from numbers import Integral

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader


WINDOW_LENGTH = 2048
SPECTRAL_BINS = 128


class GradientReverse(Function):
    """Identity forward; reverse only the encoder-side domain gradient."""
    @staticmethod
    def forward(ctx, features, strength):
        value = float(strength)
        if not math.isfinite(value) or value < 0:
            raise ValueError('GRL strength must be finite and non-negative')
        ctx.strength = value
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(features, strength):
    return GradientReverse.apply(features, strength)


def log_power_spectrum(x, window=None):
    """Centered symmetric Hann, no DC, sum each 8 powers, normalize, log."""
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3:
        raise ValueError('Spectrum requires a floating-point [N,C,2048] tensor')
    if min(x.shape[:2]) < 1 or x.shape[-1] != WINDOW_LENGTH:
        raise ValueError('Spectrum requires positive batch/channels and exactly 2048 samples')
    if not torch.isfinite(x).all():
        raise ValueError('Spectrum inputs must be finite')
    if window is None:
        window = torch.hann_window(WINDOW_LENGTH, periodic=False, dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered * window, dim=-1)[..., 1:]
    power = spectrum.abs().square().reshape(*x.shape[:2], SPECTRAL_BINS, 8).sum(dim=-1)
    relative = power / power.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    result = (relative + 1e-8).log()
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite normalized log spectrum')
    return result


def nonempty_channel_mask(batch, channels, device, generator=None):
    """Uniform draw over the nonempty subsets, independently for each sample."""
    if not 1 <= channels <= 16 or batch < 1:
        raise ValueError('Subset training supports 1 to 16 available channels')
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError('Channel mask generator must be a torch.Generator')
    if generator is not None and generator.device.type != device.type:
        raise ValueError('Channel mask generator and sensors must be on the same device type')
    codes = torch.randint(1, 2 ** channels, (batch, 1), device=device, generator=generator)
    bits = torch.arange(channels, device=device)
    return (codes.bitwise_right_shift(bits).bitwise_and(1)).bool()


@torch.no_grad()
def training_spectral_statistics(dataset, batch_size):
    """CPU-only sequential pass. Ignore labels and preserve all training RNGs.

    The caller passes the *training dataset*, never an evaluation loader. A
    dedicated generator prevents even DataLoader's base-seed draw from changing
    source sampling or model augmentation RNG. Every training row is used once,
    including a final partial batch. Original row IDs are hashed when available.
    """
    if len(dataset) < 1 or batch_size < 1:
        raise ValueError('Nonempty training data and positive batch size required')
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=0, generator=generator)
    window = torch.hann_window(WINDOW_LENGTH, periodic=False)
    count, rows, mean, m2, channels = 0, 0, None, None, None
    digest = hashlib.sha256()
    for inputs, _ignored_labels in loader:
        inputs = inputs.detach().to(device='cpu', dtype=torch.float32)
        values = log_power_spectrum(inputs, window)
        current_channels = values.shape[1]
        if channels is not None and channels != current_channels:
            raise ValueError('One calibration dataset must use one channel selection')
        channels = current_channels
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
    # Constant source bins carry no estimated variation; do not divide by zero.
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    indices = getattr(dataset, 'indices', None)
    evidence = {'rows': rows, 'channels': channels, 'sensor_windows': count,
                'feature_bytes_sha256': digest.hexdigest(), 'device': 'cpu',
                'ordered_original_indices_sha256': None if indices is None else
                hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                'labels_used': False, 'drop_last': False}
    return mean.float(), std.float(), evidence


class SpectralSharedEncoder(nn.Module):
    def __init__(self, input_channels, num_classes, mean_adaptation=False, channel_subsets=False,
                 zscore_adaptation=False, encoder_kind='mlp', domain_adversarial=False,
                 source_mask_generator=None, encoder_width=64, fusion='mean'):
        super().__init__()
        for name, value in (('input_channels', input_channels), ('num_classes', num_classes)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if channel_subsets and input_channels > 16:
            raise ValueError('Subset training supports at most 16 source channels')
        if mean_adaptation and zscore_adaptation:
            raise ValueError('Mean and z-score adaptation are mutually exclusive')
        if encoder_kind not in {'mlp', 'linear'}:
            raise ValueError('Unknown spectral encoder kind')
        if type(encoder_width) is not int or encoder_width not in (64,128):
            raise ValueError('Encoder width must be 64 or 128')
        if encoder_kind!='mlp' and encoder_width!=64:
            raise ValueError('Width ablation requires an MLP encoder')
        if domain_adversarial and encoder_kind != 'mlp':
            raise ValueError('DANN requires a shared spectral MLP')
        if fusion not in ('mean', 'attention'):
            raise ValueError('Unknown sensor fusion')
        self.fusion_mode = fusion
        self.input_channels = int(input_channels)
        self.mean_adaptation = bool(mean_adaptation)
        self.zscore_adaptation = bool(zscore_adaptation)
        self.target_statistics_adaptation = self.mean_adaptation or self.zscore_adaptation
        self.encoder_kind = encoder_kind
        self.encoder_width = encoder_width
        self.channel_subsets = bool(channel_subsets)
        if source_mask_generator is not None and not isinstance(source_mask_generator, torch.Generator):
            raise TypeError('Source mask generator must be a torch.Generator')
        if source_mask_generator is not None and not self.channel_subsets:
            raise ValueError('Source mask generator requires channel_subsets=True')
        self.source_mask_generator = source_mask_generator
        self.domain_adversarial = bool(domain_adversarial)
        self.register_buffer('hann', torch.hann_window(WINDOW_LENGTH, periodic=False))
        self.register_buffer('source_center', torch.zeros(SPECTRAL_BINS))
        self.register_buffer('source_scale', torch.ones(SPECTRAL_BINS))
        self.register_buffer('target_center', torch.zeros(SPECTRAL_BINS))
        self.register_buffer('target_scale', torch.ones(SPECTRAL_BINS))
        self.register_buffer('source_calibrated', torch.tensor(False))
        self.register_buffer('target_calibrated', torch.tensor(False))
        self.encoder = (nn.Sequential(nn.Linear(SPECTRAL_BINS, encoder_width), nn.ReLU(),
                                      nn.Linear(encoder_width, encoder_width), nn.ReLU())
                        if encoder_kind == 'mlp' else nn.Identity())
        self.classifier = nn.Linear(encoder_width if encoder_kind == 'mlp' else SPECTRAL_BINS, int(num_classes))
        self.domain_discriminator = (nn.Sequential(nn.Linear(encoder_width, 32), nn.ReLU(), nn.Linear(32, 2))
                                     if self.domain_adversarial else None)
        if fusion == 'attention':
            # No RNG draw: shared encoder/classifier initialization stays matched.
            self.fusion_score = nn.Parameter(torch.zeros(self.classifier.in_features))
        self.calibration_evidence = None

    @torch.no_grad()
    def calibrate(self, source_train, target_train=None, batch_size=32):
        if self.source_calibrated.item():
            raise ValueError('Statistics are frozen after calibration; construct a new model to refit')
        if self.target_statistics_adaptation != (target_train is not None):
            raise ValueError('Only explicit target-statistics adaptation receives target training data')
        source_mean, source_std, source_evidence = training_spectral_statistics(source_train, batch_size)
        if source_evidence['channels'] != self.input_channels:
            raise ValueError('Calibration source channels differ from the model configuration')
        target_mean, target_std = source_mean, source_std
        target_evidence = None
        if target_train is not None:
            target_mean, target_std, target_evidence = training_spectral_statistics(target_train, batch_size)
        self.source_center.copy_(source_mean)
        self.source_scale.copy_(source_std)
        self.target_center.copy_(target_mean)
        self.target_scale.copy_(target_std)
        self.source_calibrated.fill_(True)
        self.target_calibrated.fill_(target_train is not None)
        self.calibration_evidence = {'source_train': source_evidence, 'target_train': target_evidence,
                                     'source_transform': '(log_power - source_train_mean) / source_train_std',
                                     'target_transform': ('(log_power - target_train_mean) / target_train_std' if self.zscore_adaptation else
                                                          '(log_power - target_train_mean) / source_train_std' if self.mean_adaptation else
                                                          '(log_power - source_train_mean) / source_train_std'),
                                     'target_labels_used': False, 'validation_or_final_statistics_used': False,
                                     'frozen_after_calibration': True}
        return self.calibration_evidence

    def standardized_spectrum(self, x, domain):
        if domain not in {'source', 'target'}:
            raise ValueError('Unknown spectral domain')
        if not self.source_calibrated.item() or (self.target_statistics_adaptation and not self.target_calibrated.item()):
            raise RuntimeError('Train-only statistics must be calibrated before encoding')
        target_adapted = domain == 'target' and self.target_statistics_adaptation
        center = self.target_center if target_adapted else self.source_center
        scale = self.target_scale if domain == 'target' and self.zscore_adaptation else self.source_scale
        return (log_power_spectrum(x, self.hann) - center) / scale

    def fuse_sensors(self, sensors, mask=None):
        if self.fusion_mode == 'mean':
            if mask is None:
                return sensors.mean(dim=1)
            return (sensors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        if sensors.ndim != 3 or sensors.shape[1] < 1 or not torch.isfinite(sensors).all():
            raise ValueError('Attention requires finite nonempty sensor features')
        if mask is None:
            mask = torch.ones(sensors.shape[:2], dtype=torch.bool, device=sensors.device)
        if (mask.shape != sensors.shape[:2] or not torch.all((mask == 0) | (mask == 1))
                or not torch.all(mask.sum(dim=1) > 0)):
            raise ValueError('Attention mask must retain at least one sensor per row')
        mask = mask.bool()
        scores = sensors @ self.fusion_score
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite sensor attention scores')
        weights = scores.masked_fill(~mask, -float('inf')).softmax(dim=1)
        uniform = mask.to(sensors.dtype) / mask.sum(dim=1, keepdim=True)
        base = (sensors.mean(dim=1) if mask.all() else
                (sensors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True))
        # Algebraically weighted pooling. Zero scoring starts at the exact old
        # floating-point mean; C=1 always gives that single observed sensor.
        return base + ((weights - uniform).unsqueeze(-1) * sensors).sum(dim=1)

    def _encode(self, x, domain):
        spectrum = self.standardized_spectrum(x, domain)
        sensors = self.encoder(spectrum)
        if self.training and domain == 'source' and self.channel_subsets:
            if self.source_mask_generator is None:
                mask = nonempty_channel_mask(sensors.shape[0], sensors.shape[1], sensors.device)
            else:
                mask = nonempty_channel_mask(sensors.shape[0], sensors.shape[1], sensors.device,
                                             self.source_mask_generator)
            features = self.fuse_sensors(sensors, mask)
        else:
            features = self.fuse_sensors(sensors)
        return features, self.classifier(features)

    def encode_source(self, x):
        return self._encode(x, 'source')

    def encode_source_views(self, x):
        """Expose source sensor logits for an explicit source-supervision study.

        This shares encoder/classifier parameters; no additional heads, channel
        masking, target rows or target-to-source correspondences are introduced.
        Ordinary fused encoding remains unchanged for historical controls.
        """
        if self.channel_subsets:
            raise ValueError('View supervision requires unmasked source channels')
        sensors = self.encoder(self.standardized_spectrum(x, 'source'))
        features = self.fuse_sensors(sensors)
        return features, self.classifier(features), self.classifier(sensors)

    def encode_target(self, x):
        return self._encode(x, 'target')

    def predict_source_branch(self, x):
        return self.encode_source(x)[1]

    def predict_target(self, x):
        return self.encode_target(x)[1]

    def domain_logits(self, features, grl_strength):
        if self.domain_discriminator is None:
            raise RuntimeError('Domain discriminator is enabled only for DANN')
        if features.ndim != 2 or features.shape[1] != self.encoder_width:
            raise ValueError('DANN fused features must match encoder width')
        return self.domain_discriminator(gradient_reverse(features, grl_strength))

    def forward(self, source, target):
        return (*self.encode_source(source), *self.encode_target(target))

    def branch_regularization_loss(self):
        return self.classifier.weight.new_tensor(0.)

    def architecture_metadata(self):
        details = {'identity': ('Shared spectrum with DANN-style domain adversary; not original WIDAN'
                                if self.domain_adversarial else
                                'Shared spectrum candidate; known components, empirical contribution pending'),
                'input_source_channels': self.input_channels, 'input_target_channels': 'observed selected channels only',
                'window': '2048 symmetric Hann; centered; rFFT DC removed', 'spectral_bins': SPECTRAL_BINS,
                'power_pool': 'sum 8 adjacent positive-frequency bins, then unit-sum power and log(p+1e-8)',
                'spectrum_precision': 'float32 torch; CPU probe used float64 numpy',
                'encoder': (f'shared 128-{self.encoder_width}-ReLU-{self.encoder_width}-ReLU MLP' if self.encoder_kind == 'mlp' else
                            'identity spectral encoder with linear classifier'),
                'encoder_kind': self.encoder_kind, 'fusion': 'arithmetic mean after sensor encoding',
                'normalization': 'fixed source-train per-frequency population std over rows and observed channels',
                'mean_adaptation': self.mean_adaptation, 'zscore_adaptation': self.zscore_adaptation,
                'source_channel_subsets': self.channel_subsets,
                'subset_policy': 'uniform nonempty subsets per source sample during training only' if self.channel_subsets else 'disabled',
                'parameter_count': sum(p.numel() for p in self.parameters()), 'calibration': self.calibration_evidence}
        if self.domain_adversarial:
            details['domain_adversary'] = {
                'source_domain_id': 0, 'target_domain_id': 1,
                'fused_feature_dim': self.encoder_width, 'discriminator': f'{self.encoder_width}-32-ReLU-2',
                'gradient_reversal': 'identity forward; negative scheduled strength backward',
                'target_fault_labels_used': False,
                'discriminator_parameter_count': sum(p.numel() for p in self.domain_discriminator.parameters())}
        if self.encoder_width!=64:
            details['encoder_width']=self.encoder_width
        if self.fusion_mode == 'attention':
            details.update(fusion_mode='attention', fusion='softmax linear content score after sensor encoding',
                           fusion_parameter_count=self.fusion_score.numel(), fusion_initialization='zeros',
                           single_sensor_fusion='identity; gain can only arise through trained shared representation')
        if self.channel_subsets and self.source_mask_generator is not None:
            details['subset_rng'] = 'independent_torch_generator; state not part of model weights'
        return details
