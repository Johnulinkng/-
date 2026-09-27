"""Independent schema-3 provenance audit and post-selection target-dev evaluation.

Run this outside the training process. The final split is never indexed as a
signal or label array by the evaluator. A private preflight hashes complete
files, including final bytes, without decoding final rows; the private audit
also has same-record class mappings, so this is a code/process boundary rather
than information-theoretic secrecy.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from antipair_builder import PARTITIONS, _public_row, file_sha256
from public_loader import (PublicProtocolError, _inside, _json, _npy_header,
                           load_public_task)


def _private_file(root, value):
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PublicProtocolError('Private audit path escapes private_root')
    return path


def _validate_origins(public_root, private_root, plan, version, private_audit, task_id):
    refs = plan.get('reference_sha256', {})
    if set(refs) != {'inventory.json', 'raw_records.json'}:
        raise PublicProtocolError('Reference snapshot SHA set is incomplete')
    for name, expected in refs.items():
        path = private_root / 'reference' / name
        if file_sha256(path) != expected:
            raise PublicProtocolError(f'Private reference snapshot {name} SHA mismatch')
    records = _json(private_root / 'reference' / 'raw_records.json')
    expected_record_index = sorted(
        ({'condition': record['condition'], 'label': record['label'],
          'relative_path': record['relative_path'], 'sha256': record['sha256']}
         for record in records), key=lambda record: (record['condition'], record['label']))
    if private_audit.get('raw_records') != expected_record_index:
        raise PublicProtocolError('Private audit raw-record map differs from snapshot')
    by_id = {record['sha256']: record for record in records}
    if len(by_id) != len(records):
        raise PublicProtocolError('Raw record SHA reused across classes or conditions')
    banks = private_audit.get('bank_origins')
    public_banks = version.get('bank_file_sha256', {})
    if not isinstance(banks, dict) or set(banks) != set(public_banks):
        raise PublicProtocolError('Private/public bank set mismatch')
    token_salt = private_audit.get('token_salt')
    if not isinstance(token_salt, str) or len(token_salt) < 32:
        raise PublicProtocolError('Private row-token salt missing')
    quotas, window, stride, gap = (plan['quotas_per_class'], plan['window_size'],
                                    plan['stride_samples'], plan['gap_samples'])
    classes = plan['classes']
    role_banks = {}
    for spec in plan['role_banks']:
        condition, group, role = spec['condition'], spec['group'], spec['role']
        bank_id = f'{condition}_{group}_{role}'
        info = banks[bank_id]
        internal = info.get('rows')
        permutation = info.get('canonical_to_bank_index')
        if not isinstance(internal, list) or not isinstance(permutation, list):
            raise PublicProtocolError(f'{bank_id}: private row/permutation list missing')
        nrows = classes * sum(quotas[role].values())
        if (len(internal) != nrows or len(permutation) != nrows or
                sorted(permutation) != list(range(nrows))):
            raise PublicProtocolError(f'{bank_id}: private row count/permutation invalid')
        descriptor = public_banks[bank_id]['row_provenance.json']
        public_path = _inside(public_root, descriptor['path'])
        public_rows = _json(public_path)
        if len(public_rows) != nrows:
            raise PublicProtocolError(f'{bank_id}: public/private provenance count mismatch')
        counts = Counter()
        intervals = defaultdict(list)
        for index, row in enumerate(internal):
            if public_rows[index] != _public_row(row, role, token_salt):
                raise PublicProtocolError(f'{bank_id}: public blind row does not match private origin')
            record = by_id.get(row.get('record_id'))
            if record is None or row.get('condition') != condition or record['condition'] != condition:
                raise PublicProtocolError(f'{bank_id}: origin record/condition mismatch')
            label, part = row.get('label'), row.get('partition')
            if (label != record['label'] or label not in range(classes) or part not in PARTITIONS):
                raise PublicProtocolError(f'{bank_id}: origin class/partition mismatch')
            start, end = row.get('start'), row.get('end')
            length = record['shape'][0]
            if (not isinstance(start, int) or not isinstance(end, int) or
                    start < 0 or end - start != window or end > length or
                    row.get('center_sample') != start + window // 2):
                raise PublicProtocolError(f'{bank_id}: raw window interval invalid')
            b1, b2 = 55 * length // 100, 85 * length // 100
            bounds = {'train': (0, b1 - gap), 'dev': (b1, b2 - gap),
                      'final': (b2, length)}
            left, right = bounds[part]
            if start < left or end > right:
                raise PublicProtocolError(f'{bank_id}: origin crosses macro partition')
            counts[(label, part)] += 1
            intervals[record['sha256']].append((start, end, role, part))
        if any(counts[(label, part)] != quotas[role][part]
               for label in range(classes) for part in PARTITIONS):
            raise PublicProtocolError(f'{bank_id}: fixed per-class/partition quota mismatch')
        for record_id, entries in intervals.items():
            ordered = sorted(entries)
            for left, right in zip(ordered, ordered[1:]):
                if right[0] - left[0] < stride or right[0] < left[1]:
                    raise PublicProtocolError(f'{bank_id}: within-role windows overlap/break stride')
                if left[3] != right[3] and right[0] - left[1] < gap:
                    raise PublicProtocolError(f'{bank_id}: partition embargo violated')
        role_banks[bank_id] = internal
    for task in _json(private_root / 'reference' / 'inventory.json'):
        if task['task_id'] not in plan['task_ids']:
            continue
        source = task['source']
        target = task['target']
        source_id = f'{source["condition"]}_{source["group"]}_source'
        target_id = f'{target["condition"]}_{target["group"]}_target'
        if source_id not in role_banks or target_id not in role_banks:
            raise PublicProtocolError('Task references absent private role bank')
        rows_by_record = defaultdict(list)
        seen_roles = defaultdict(set)
        for role, bank_id in (('source', source_id), ('target', target_id)):
            for row in role_banks[bank_id]:
                record_id = row['record_id']
                rows_by_record[record_id].append((row['start'], row['end'], role, row['partition']))
                seen_roles[record_id].add(role)
        minimum = None
        for record_id, entries in rows_by_record.items():
            ordered = sorted(entries)
            for left, right in zip(ordered, ordered[1:]):
                separation = right[0] - left[1]
                if left[2] != right[2]:
                    minimum = separation if minimum is None else min(minimum, separation)
                    if separation < gap:
                        raise PublicProtocolError(f'{task["task_id"]}: cross-role embargo violated')
                elif separation < 0:
                    raise PublicProtocolError(f'{task["task_id"]}: same-role raw overlap')
                if left[3] != right[3] and separation < gap:
                    raise PublicProtocolError(f'{task["task_id"]}: partition embargo violated')
            if len(seen_roles[record_id]) == 2:
                for part in PARTITIONS:
                    first = [item for item in entries if item[2] == 'source' and item[3] == part]
                    second = [item for item in entries if item[2] == 'target' and item[3] == part]
                    if not first or not second:
                        raise PublicProtocolError(f'{task["task_id"]}: role partition missing')
                    if plan['role_order'] == 'target_first':
                        first, second = second, first
                    if max(item[1] for item in first) + gap > min(item[0] for item in second):
                        raise PublicProtocolError(f'{task["task_id"]}: mirror role order violated')
        checked = {'common_record_count': sum(len(value) == 2 for value in seen_roles.values()),
                   'minimum_cross_role_gap_samples': minimum, 'all_roles_checked': True}
        manifest = _json(public_root / 'tasks' / task['task_id'] / 'task_manifest.json')
        if (manifest.get('source') != task['source'] or
                manifest.get('target') != task['target'] or
                manifest.get('task_type') != task['task_type'] or
                manifest.get('cross_role_audit') != checked or
                plan.get('task_origin_audit', {}).get(task['task_id']) != checked or
                private_audit.get('task_origin_audit', {}).get(task['task_id']) != checked):
            raise PublicProtocolError(f'{task["task_id"]}: audit summary mismatch')
    return role_banks


def _preflight(public_root, private_root, task_id, channels):
    public_root = Path(public_root).resolve(strict=True)
    private_root = Path(private_root).resolve(strict=True)
    if private_root != public_root.parent / f'{public_root.name}_private_audit':
        raise PublicProtocolError('Expected sibling private audit directory')
    inputs = load_public_task(public_root, task_id, channels, role='evaluate_dev')
    plan = _json(public_root / 'build_plan.json')
    version = _json(public_root / 'version_manifest.json')
    manifest = _json(public_root / 'tasks' / task_id / 'task_manifest.json')
    private_path = private_root / 'private_audit.json'
    private_sha = file_sha256(private_path)
    if private_sha != version.get('private_audit_sha256'):
        raise PublicProtocolError('Private audit SHA differs from public version receipt')
    private_audit = _json(private_path)
    if private_audit.get('plan_sha256') != plan['plan_sha256']:
        raise PublicProtocolError('Private audit plan SHA mismatch')
    for bank_id, files in version['bank_file_sha256'].items():
        for name, descriptor in files.items():
            path = _inside(public_root, descriptor['path'])
            if file_sha256(path) != descriptor['sha256']:
                raise PublicProtocolError(f'{bank_id}/{name}: full public file SHA mismatch')
    private_target = private_audit.get('target_label_files', {})
    for bank_id, descriptor in private_target.items():
        path = _private_file(private_root, descriptor['path'])
        if file_sha256(path) != descriptor['sha256']:
            raise PublicProtocolError(f'{bank_id}: private target-label file SHA mismatch')
    _validate_origins(public_root, private_root, plan, version, private_audit, task_id)
    target = manifest['target']
    bank_id = f'{target["condition"]}_{target["group"]}_target'
    if (bank_id not in private_target or
            manifest['target_label_sha256_for_sealed_evaluation'] != private_target[bank_id]['sha256']):
        raise PublicProtocolError('Task target-label seal differs from private file')
    receipt = {'task_id': task_id, 'plan_sha256': plan['plan_sha256'],
               'task_manifest_sha256': inputs.manifest_sha256,
               'private_audit_sha256': private_sha,
               'full_public_bank_sha_verified': True,
               'private_origin_embargo_verified': True,
               'final_signal_or_label_rows_decoded': False}
    target_origins = private_audit['bank_origins'][bank_id]['rows']
    return (receipt, inputs, manifest,
            _private_file(private_root, private_target[bank_id]['path']), target_origins)


def verify_private_audit(public_root, private_root, task_id, channels):
    """Read-only offline preflight. This hashes full files but decodes no final rows."""
    try:
        receipt, _, _, _, _ = _preflight(public_root, private_root, task_id, channels)
        return receipt
    except (KeyError, IndexError, TypeError, OSError) as exc:
        raise PublicProtocolError('Malformed private audit or task metadata') from exc


def _metrics(truth, prediction, classes):
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for true, predicted in zip(truth, prediction):
        confusion[int(true), int(predicted)] += 1
    recall = []
    f1 = []
    for label in range(classes):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        recall.append(tp / support if support else 0.0)
        denom = support + predicted
        f1.append(2 * tp / denom if denom else 0.0)
    return {'accuracy': float(np.trace(confusion) / len(truth)),
            'macro_f1': float(np.mean(f1)),
            'per_class_recall': recall, 'confusion_matrix': confusion.tolist(),
            'support_per_class': confusion.sum(axis=1).tolist()}


def verify_private_and_evaluate(public_root, private_root, task_id, channels,
                                locked_checkpoint, expected_checkpoint_sha256,
                                predict_fn, batch_size=64):
    """Evaluate target-dev once with an already selected, SHA-locked checkpoint.

    ``predict_fn(checkpoint_path, float32_batch)`` receives no private root or
    labels and must return one integer class per row. Final rows are never
    indexed. This function performs no checkpoint selection.
    """
    checkpoint = Path(locked_checkpoint).resolve(strict=True)
    actual_sha = file_sha256(checkpoint)
    if (not isinstance(expected_checkpoint_sha256, str) or
            actual_sha != expected_checkpoint_sha256):
        raise PublicProtocolError('Locked checkpoint SHA mismatch')
    if not callable(predict_fn) or not isinstance(batch_size, int) or batch_size <= 0:
        raise PublicProtocolError('Need a predictor and positive batch size')
    try:
        receipt, inputs, manifest, label_path, target_origins = _preflight(
            public_root, private_root, task_id, channels)
        predictions = []
        for start in range(0, len(inputs.target_dev), batch_size):
            end = min(start + batch_size, len(inputs.target_dev))
            batch = np.stack([inputs.target_dev[i][0] for i in range(start, end)])
            value = np.asarray(predict_fn(checkpoint, batch))
            if (value.shape != (end - start,) or value.dtype.kind not in 'iu' or
                    np.any(value < 0) or np.any(value >= inputs.num_classes)):
                raise PublicProtocolError('Predictor returned invalid class IDs')
            predictions.extend(value.tolist())
        # Only after predictions are fixed do we decode private target-dev labels.
        labels = _npy_header(label_path, dtype=np.int64)
        indices = manifest['splits']['target_test']['indices']
        truth = np.asarray(labels[indices])
        expected_truth = np.asarray([target_origins[i]['label'] for i in indices])
        if (len(truth) != len(predictions) or np.any(truth < 0) or
                np.any(truth >= inputs.num_classes) or
                not np.array_equal(truth, expected_truth)):
            raise PublicProtocolError('Private target-dev labels invalid')
        if file_sha256(checkpoint) != expected_checkpoint_sha256:
            raise PublicProtocolError('Locked checkpoint changed during evaluation')
        result = {'receipt': receipt, 'checkpoint_sha256': actual_sha,
                  'evaluated_split': 'target_dev_only', 'n_samples': len(truth)}
        result.update(_metrics(truth, predictions, inputs.num_classes))
        return result
    except (KeyError, IndexError, TypeError, OSError) as exc:
        raise PublicProtocolError('Malformed private audit or evaluation input') from exc
