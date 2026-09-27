"""Schema-3 public access. Training never receives private paths or target labels.

Only headers and metadata hashes are checked here. Hashing a complete target
data.npy would read final signal bytes in the training process; full file and
raw-interval verification belongs to private_evaluator.verify_private_audit.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from antipair_builder import PARTITIONS, file_sha256, json_sha256


EXPECTED_PROTOCOL = 'antipaired_role_banks_v1'
SPLITS = {'source_train': ('source', 'train'),
          'source_val': ('source', 'dev'),
          'target_train': ('target', 'train'),
          'target_test': ('target', 'dev'),
          'target_final_test': ('target', 'final')}


class PublicProtocolError(ValueError):
    """Public task metadata is incompatible, incomplete, or internally inconsistent."""


class SplitDataset:
    """Read-only indexed view; no raw bank or held-out indices are public API."""

    __slots__ = ('_data', '_labels', '_indices', '_channels', '_target')

    def __init__(self, data, labels, indices, channels, target):
        self._data = data
        self._labels = labels
        self._indices = tuple(indices)
        self._channels = tuple(channels)
        self._target = target

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError('Dataset index must be an integer')
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self._indices[index]
        signal = np.asarray(self._data[row, self._channels, :], dtype=np.float32).copy()
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)
        signal = ((signal - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
        label = -1 if self._target else int(self._labels[row])
        return signal, label

    def log_ac_rms(self, index):
        """Optional pre-normalization amplitude, one scalar per allowed channel.

        Does not read labels, other channels or expose bank indices. Default
        __getitem__ preprocessing and its target-label mask remain unchanged.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError('Dataset index must be an integer')
        if index < 0 or index >= len(self):
            raise IndexError(index)
        signal = np.asarray(self._data[self._indices[index], self._channels, :], dtype=np.float64).copy()
        if not np.isfinite(signal).all():
            raise ValueError('AC RMS requires finite raw samples')
        centered = signal - signal.mean(axis=-1, keepdims=True)
        rms = np.sqrt(np.mean(centered * centered, axis=-1))
        if not np.all(np.isfinite(rms) & (rms > 0)):
            raise ValueError('AC RMS requires positive nonconstant signal energy')
        return np.log(rms)


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    dataset: str
    plan_sha256: str
    manifest_sha256: str
    window_size: int
    num_classes: int
    channels: dict
    source_train: SplitDataset
    source_val: SplitDataset
    target_train: SplitDataset


@dataclass(frozen=True)
class SourceOnlyTask:
    task_id: str
    dataset: str
    plan_sha256: str
    manifest_sha256: str
    window_size: int
    num_classes: int
    channels: dict
    source_train: SplitDataset
    source_val: SplitDataset


@dataclass(frozen=True)
class EvaluationInputs:
    task_id: str
    dataset: str
    plan_sha256: str
    manifest_sha256: str
    window_size: int
    num_classes: int
    channels: dict
    target_dev: SplitDataset


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicProtocolError(f'Cannot read public JSON: {path}') from exc


def _inside(root, path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PublicProtocolError('Public file points outside public_root')
    return resolved


def _npy_header(path, shape=None, dtype=None):
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PublicProtocolError(f'Invalid NPY header: {path}') from exc
    if not isinstance(array, np.memmap) or (shape is not None and tuple(array.shape) != tuple(shape)):
        raise PublicProtocolError(f'Unexpected NPY shape: {path}')
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise PublicProtocolError(f'Unexpected NPY dtype: {path}')
    expected_size = array.offset + array.size * array.dtype.itemsize
    if Path(path).stat().st_size != expected_size:
        raise PublicProtocolError(f'NPY file size differs from header: {path}')
    return array


def _channels(channels, shapes, manifest):
    if not isinstance(channels, dict) or set(channels) != {'source', 'target'}:
        raise PublicProtocolError('channels must map source and target to local slot lists')
    result = {}
    for role in ('source', 'target'):
        slots = channels[role]
        if (not isinstance(slots, (list, tuple)) or
                len(slots) not in (1, 3) or
                any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in slots) or
                len(set(slots)) != len(slots) or max(slots, default=-1) >= shapes[role][1]):
            raise PublicProtocolError(f'Invalid {role} channel slots')
        expected_stored = len(manifest[role]['raw_columns_zero_based'])
        if shapes[role][1] != expected_stored:
            raise PublicProtocolError(f'{role} NPY channel count differs from manifest')
        result[role] = tuple(slots)
    return result


def _rows(rows, expected_count, role):
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise PublicProtocolError(f'{role} public provenance count differs from bank')
    tokens = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'row_token', 'partition'}:
            raise PublicProtocolError(f'{role} provenance exposes extra fields or lacks token')
        token = row['row_token']
        if (not isinstance(token, str) or len(token) != 64 or
                any(char not in '0123456789abcdef' for char in token) or token in tokens or
                row['partition'] not in PARTITIONS):
            raise PublicProtocolError(f'{role} provenance has invalid or repeated token')
        tokens.add(token)
    return tokens


def _validate_splits(manifest, plan, source_rows, target_rows, labels, num_classes, role):
    splits = manifest.get('splits')
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise PublicProtocolError('Missing or extra split definitions')
    quotas = plan['quotas_per_class']
    indices_by_split = {}
    for name, (domain, part) in SPLITS.items():
        payload = splits[name]
        rows = source_rows if domain == 'source' else target_rows
        indices = payload.get('indices') if isinstance(payload, dict) else None
        if (payload.get('domain') != domain or not isinstance(indices, list) or
                any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(rows)
                    for i in indices) or len(set(indices)) != len(indices)):
            raise PublicProtocolError(f'Invalid {name} indices')
        exact = [i for i, row in enumerate(rows) if row['partition'] == part]
        if indices != exact or len(indices) != num_classes * quotas[domain][part]:
            raise PublicProtocolError(f'{name} differs from provenance or quota')
        indices_by_split[name] = tuple(indices)
    if manifest.get('reserved_source_final_indices') != [
            i for i, row in enumerate(source_rows) if row['partition'] == 'final']:
        raise PublicProtocolError('Reserved source-final indices inconsistent')
    if labels is not None:
        for name, part in (('source_train', 'train'), ('source_val', 'dev')):
            selected = np.asarray(labels[list(indices_by_split[name])])
            if (selected.size != num_classes * quotas['source'][part] or
                    np.any(selected < 0) or np.any(selected >= num_classes) or
                    not np.array_equal(np.bincount(selected, minlength=num_classes),
                                       np.full(num_classes, quotas['source'][part]))):
                raise PublicProtocolError(f'{name} source labels violate closed-set quota')
    audit = manifest.get('cross_role_audit')
    if (not isinstance(audit, dict) or audit.get('all_roles_checked') is not True or
            not isinstance(audit.get('common_record_count'), int) or
            audit['common_record_count'] < 0):
        raise PublicProtocolError('Missing private cross-role audit summary')
    minimum = audit.get('minimum_cross_role_gap_samples')
    if minimum is not None and (not isinstance(minimum, int) or minimum < plan['gap_samples']):
        raise PublicProtocolError('Private audit summary reports insufficient embargo')
    return indices_by_split


def _read_public(public_root, task_id, channels, role):
    if role not in ('source_only', 'train', 'evaluate_dev'):
        raise PublicProtocolError('role must be source_only, train, or evaluate_dev')
    root = Path(public_root).resolve(strict=True)
    version = _json(root / 'version_manifest.json')
    plan = _json(root / 'build_plan.json')
    manifest_path = root / 'tasks' / task_id / 'task_manifest.json'
    manifest = _json(manifest_path)
    if any(item.get('schema_version') != 3 or item.get('protocol') != EXPECTED_PROTOCOL
           for item in (version, plan, manifest)):
        raise PublicProtocolError('Expected schema 3 anti-paired protocol')
    if any(item.get('loader_compatible') is not False for item in (version, plan, manifest)):
        raise PublicProtocolError('Schema 3 must declare old-loader incompatibility')
    if task_id != manifest.get('task_id') or task_id not in plan.get('task_ids', ()):
        raise PublicProtocolError('Task not included in public plan')
    plan_value = {k: value for k, value in plan.items()
                  if k not in ('plan_sha256', 'builder_code_sha256')}
    if (json_sha256(plan_value) != plan.get('plan_sha256') or
            plan['plan_sha256'] != version.get('plan_sha256') or
            plan['plan_sha256'] != manifest.get('plan_sha256')):
        raise PublicProtocolError('Public plan SHA mismatch')
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != version.get('task_manifest_sha256', {}).get(task_id):
        raise PublicProtocolError('Task manifest SHA mismatch')
    code_path = root / 'code' / 'antipair_builder.py'
    if file_sha256(code_path) != version.get('builder_code_sha256'):
        raise PublicProtocolError('Builder code snapshot SHA mismatch')
    if (manifest.get('classes') != plan.get('classes') or
            manifest.get('window_size') != plan.get('window_size') or
            manifest.get('stride_samples') != plan.get('stride_samples') or
            manifest.get('gap_samples') != plan.get('gap_samples') or
            manifest.get('quotas_per_class') != plan.get('quotas_per_class') or
            manifest.get('role_order') != plan.get('role_order') or
            manifest.get('dataset') != plan.get('dataset')):
        raise PublicProtocolError('Task configuration differs from plan')
    num_classes, window = plan['classes'], plan['window_size']
    if not isinstance(num_classes, int) or num_classes < 2 or not isinstance(window, int) or window <= 0:
        raise PublicProtocolError('Invalid class count or window length')
    if set(manifest.get('data_files', {})) != {'source_data', 'source_label', 'target_data'}:
        raise PublicProtocolError('Training-visible manifest must not include target label path')
    paths = {}
    descriptors = {}
    for role_name in ('source', 'target'):
        side = manifest[role_name]
        bank_id = f'{side["condition"]}_{side["group"]}_{role_name}'
        bank_files = version.get('bank_file_sha256', {}).get(bank_id)
        if not isinstance(bank_files, dict):
            raise PublicProtocolError(f'Bank {bank_id} missing from version manifest')
        for stem in (('data.npy', 'row_provenance.json', 'label.npy')
                     if role_name == 'source' else ('data.npy', 'row_provenance.json')):
            item = bank_files.get(stem)
            if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
                raise PublicProtocolError(f'Bank {bank_id} file descriptor invalid')
            expected = root / 'domains' / bank_id / stem
            actual = _inside(root, item['path'])
            if actual != expected.resolve(strict=True):
                raise PublicProtocolError(f'Bank {bank_id} file path differs from expected role bank')
            paths[(role_name, stem)] = (actual, item['sha256'])
            descriptors[(role_name, stem)] = item
    for field, key in (('source_data', ('source', 'data.npy')),
                       ('source_label', ('source', 'label.npy')),
                       ('target_data', ('target', 'data.npy'))):
        # Keep the original descriptor spelling (including Windows short-path
        # aliases). _inside/expected above already verify the canonical path.
        if manifest['data_files'][field] != descriptors[key]:
            raise PublicProtocolError(f'{field} descriptor differs from version manifest')
    if not isinstance(manifest.get('target_label_sha256_for_sealed_evaluation'), str):
        raise PublicProtocolError('Target-label sealed hash missing')
    rows = {}
    for domain in ('source', 'target'):
        path, expected_sha = paths[(domain, 'row_provenance.json')]
        if file_sha256(path) != expected_sha:
            raise PublicProtocolError(f'{domain} provenance SHA mismatch')
        rows[domain] = _json(path)
        if manifest.get('row_provenance', {}).get(domain) != rows[domain]:
            raise PublicProtocolError(f'{domain} task provenance differs from bank')
    source_data = _npy_header(paths[('source', 'data.npy')][0])
    target_data = _npy_header(paths[('target', 'data.npy')][0])
    source_labels = _npy_header(paths[('source', 'label.npy')][0],
                                (source_data.shape[0],), np.int64)
    if file_sha256(paths[('source', 'label.npy')][0]) != paths[('source', 'label.npy')][1]:
        raise PublicProtocolError('Source label SHA mismatch')
    if (source_data.dtype != np.float32 or target_data.dtype != np.float32 or
            source_data.ndim != 3 or target_data.ndim != 3 or
            source_data.shape[2] != window or target_data.shape[2] != window):
        raise PublicProtocolError('Public signal array shape/dtype differs from plan')
    selected_channels = _channels(channels,
                                  {'source': source_data.shape, 'target': target_data.shape}, manifest)
    source_tokens = _rows(rows['source'], source_data.shape[0], 'source')
    target_tokens = _rows(rows['target'], target_data.shape[0], 'target')
    if source_tokens & target_tokens:
        raise PublicProtocolError('Source and target row tokens are joinable')
    selected = _validate_splits(manifest, plan, rows['source'], rows['target'],
                                source_labels if role in ('source_only', 'train') else None,
                                num_classes, role)
    identity = dict(task_id=task_id, dataset=plan['dataset'],
                    plan_sha256=plan['plan_sha256'], manifest_sha256=manifest_sha,
                    window_size=window, num_classes=num_classes, channels=selected_channels)
    if role == 'source_only':
        return SourceOnlyTask(**identity,
                              source_train=SplitDataset(source_data, source_labels,
                                                        selected['source_train'], selected_channels['source'], False),
                              source_val=SplitDataset(source_data, source_labels,
                                                      selected['source_val'], selected_channels['source'], False))
    if role == 'train':
        return PublicTask(**identity,
                          source_train=SplitDataset(source_data, source_labels,
                                                    selected['source_train'], selected_channels['source'], False),
                          source_val=SplitDataset(source_data, source_labels,
                                                  selected['source_val'], selected_channels['source'], False),
                          target_train=SplitDataset(target_data, None,
                                                    selected['target_train'], selected_channels['target'], True))
    return EvaluationInputs(**identity,
                            target_dev=SplitDataset(target_data, None,
                                                    selected['target_test'], selected_channels['target'], True))


def load_public_task(public_root, task_id, channels, role='train'):
    """Return only the role-appropriate indexed datasets; never a final accessor.

    ``channels`` uses stored 0-based slots. Source and target each select one
    or three channels; the training arm constrains 3→3, 3→1, or 1→1.
    """
    try:
        return _read_public(public_root, task_id, channels, role)
    except (KeyError, IndexError, TypeError, OSError) as exc:
        raise PublicProtocolError('Malformed public task metadata') from exc
