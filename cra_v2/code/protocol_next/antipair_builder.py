"""Independent, label-sealed prototype for role-disjoint raw-time transfer tasks.

This writes schema 3. The existing workbench loader does not accept schema 3;
training requires a separate, reviewed loader adaptation. ``--dry-run`` only
reads JSON metadata. ``--build`` reads raw MAT recordings, never an existing
target-label array, and is deliberately not invoked by this prototype study.
"""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil

import numpy as np
from scipy.io import loadmat, whosmat


PARTITIONS = ('train', 'dev', 'final')
ROLES = ('source', 'target')
DEFAULTS = {
    'seu_gearbox': {
        'stride': 2048,
        'quotas': {'source': {'train': 120, 'dev': 60, 'final': 30},
                   'target': {'train': 120, 'dev': 60, 'final': 30}},
    },
    'gearbox': {
        'stride': 2048,
        'quotas': {'source': {'train': 70, 'dev': 25, 'final': 5},
                   'target': {'train': 50, 'dev': 25, 'final': 20}},
    },
    'waterpump': {
        'stride': 4096,
        'quotas': {'source': {'train': 300, 'dev': 150, 'final': 20},
                   'target': {'train': 300, 'dev': 150, 'final': 140}},
    },
}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def json_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def validate_config(window, stride, gap, quotas, role_order):
    if role_order not in {'source_first', 'target_first'}:
        raise ValueError('role_order must be source_first or target_first')
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
           for x in (window, stride, gap)) or stride < window or gap < window:
        raise ValueError('Require positive integer window <= stride and window <= gap')
    if set(quotas) != set(ROLES) or any(set(quotas[role]) != set(PARTITIONS) for role in ROLES):
        raise ValueError('Quotas require source and target train/dev/final roles')
    if any(isinstance(quotas[role][part], bool) or not isinstance(quotas[role][part], int)
           or quotas[role][part] <= 0 for role in ROLES for part in PARTITIONS):
        raise ValueError('Every role/partition quota must be a positive integer')


def _select_starts(start, stop, window, stride, count):
    capacity = max(0, (stop - start - window) // stride + 1)
    if capacity < count:
        raise ValueError(f'Only {capacity} start positions for {count} requested windows')
    if count == 1:
        positions = [capacity // 2]
    else:
        positions = [i * (capacity - 1) // (count - 1) for i in range(count)]
    return [start + position * stride for position in positions], capacity


def plan_record(record, window, stride, gap, quotas, role_order):
    """Return all six role/partition origins for one class recording."""
    validate_config(window, stride, gap, quotas, role_order)
    length = record['shape'][0]
    if not isinstance(length, int) or length < 1:
        raise ValueError('Raw record requires a positive integer sample length')
    b1, b2 = 55 * length // 100, 85 * length // 100
    macro = {'train': (0, b1 - gap), 'dev': (b1, b2 - gap), 'final': (b2, length)}
    first, second = (('source', 'target') if role_order == 'source_first'
                     else ('target', 'source'))
    rows = {role: {part: [] for part in PARTITIONS} for role in ROLES}
    capacity = {role: {} for role in ROLES}
    zones = {role: {} for role in ROLES}
    for part in PARTITIONS:
        left, right = macro[part]
        usable = right - left - gap
        n_first, n_second = quotas[first][part], quotas[second][part]
        if usable < 0:
            raise ValueError(f'{record["condition"]} class {record["label"]}: {part} block shorter than embargo')
        cut = left + usable * n_first // (n_first + n_second)
        zones[first][part] = (left, cut)
        zones[second][part] = (cut + gap, right)
        for role in ROLES:
            zone_start, zone_end = zones[role][part]
            starts, capacity[role][part] = _select_starts(
                zone_start, zone_end, window, stride, quotas[role][part])
            rows[role][part] = [
                {'record_id': record['sha256'], 'condition': record['condition'],
                 'label': record['label'], 'start': start, 'end': start + window,
                 'center_sample': start + window // 2, 'partition': part}
                for start in starts]
    result = {'record_id': record['sha256'], 'condition': record['condition'],
              'label': record['label'], 'length': length, 'macro_bounds': macro,
              'zones': zones, 'capacity': capacity, 'rows': rows}
    validate_record_plan(result, window, stride, gap, quotas)
    return result


def validate_record_plan(plan, window, stride, gap, quotas):
    """Fail closed on count, origin, partition and every cross-role raw interval."""
    if set(plan['rows']) != set(ROLES):
        raise ValueError('Missing role origins')
    merged = []
    for role in ROLES:
        if set(plan['rows'][role]) != set(PARTITIONS):
            raise ValueError('Missing partition origins')
        for part in PARTITIONS:
            current = plan['rows'][role][part]
            if len(current) != quotas[role][part]:
                raise ValueError('Role/partition count differs from fixed quota')
            left, right = plan['zones'][role][part]
            macro_left, macro_right = plan['macro_bounds'][part]
            if left < macro_left or right > macro_right or left > right:
                raise ValueError('Role zone crosses a macro partition')
            starts = []
            for row in current:
                start, end = row['start'], row['end']
                if (row['record_id'] != plan['record_id'] or row['partition'] != part
                        or row['condition'] != plan['condition'] or row['label'] != plan['label']
                        or not left <= start < end <= right or end - start != window
                        or row['center_sample'] != start + window // 2):
                    raise ValueError('Window origin violates its role zone')
                starts.append(start)
                merged.append((start, end, role, part))
            if starts != sorted(set(starts)) or any(b - a < stride for a, b in zip(starts, starts[1:])):
                raise ValueError('Within-role starts overlap or break the minimum stride')
    merged.sort()
    minimum_cross_role_gap = None
    for left, right in zip(merged, merged[1:]):
        separation = right[0] - left[1]
        if left[2] != right[2]:
            minimum_cross_role_gap = (separation if minimum_cross_role_gap is None
                                      else min(minimum_cross_role_gap, separation))
            if separation < gap:
                raise ValueError('Cross-role raw intervals overlap or violate embargo')
        elif separation < 0:
            raise ValueError('Within-role raw windows overlap')
        if left[3] != right[3] and separation < gap:
            raise ValueError('Cross-partition raw intervals violate embargo')
    if minimum_cross_role_gap is None:
        raise ValueError('Both role banks must be present in each recording')
    return minimum_cross_role_gap


def validate_task_origins(source_rows, target_rows, window, stride, gap, quotas, classes):
    """Recheck all source/target roles, including different partitions, per record."""
    per_record = {}
    for role, rows in (('source', source_rows), ('target', target_rows)):
        for label in range(classes):
            for part in PARTITIONS:
                count = sum(row['label'] == label and row['partition'] == part for row in rows)
                if count != quotas[role][part]:
                    raise ValueError(f'{role} class {label} {part}: count differs from fixed quota')
        for row in rows:
            if (row['label'] not in range(classes) or row['end'] - row['start'] != window
                    or row['center_sample'] != row['start'] + window // 2):
                raise ValueError('Task row has invalid label or raw interval')
            per_record.setdefault(row['record_id'], {r: [] for r in ROLES})[role].append(row)
    minimum = None
    for record_id, roles in per_record.items():
        for role in ROLES:
            for part in PARTITIONS:
                selected = sorted((row['start'], row['end']) for row in roles[role]
                                  if row['partition'] == part)
                if any(b[0] - a[0] < stride or b[0] < a[1]
                       for a, b in zip(selected, selected[1:])):
                    raise ValueError(f'{record_id}: within-role windows overlap or break stride')
        merged = sorted((row['start'], row['end'], role, row['partition'])
                        for role in ROLES for row in roles[role])
        for left, right in zip(merged, merged[1:]):
            separation = right[0] - left[1]
            if left[2] != right[2]:
                minimum = separation if minimum is None else min(minimum, separation)
                if separation < gap:
                    raise ValueError(f'{record_id}: cross-role raw intervals violate embargo')
            elif separation < 0:
                raise ValueError(f'{record_id}: within-role windows overlap')
            if left[3] != right[3] and separation < gap:
                raise ValueError(f'{record_id}: cross-partition raw intervals violate embargo')
    return {'common_record_count': sum(bool(roles['source']) and bool(roles['target'])
                                       for roles in per_record.values()),
            'minimum_cross_role_gap_samples': minimum,
            'all_roles_checked': True}


def prepare(reference_root, raw_root, dataset, role_order='source_first',
            window=2048, gap=24000, stride=None, quotas=None, task_ids=None):
    """Metadata-only capacity/preflight; never load a MAT or target label array."""
    if dataset not in DEFAULTS:
        raise ValueError('Unknown dataset')
    stride = DEFAULTS[dataset]['stride'] if stride is None else stride
    quotas = DEFAULTS[dataset]['quotas'] if quotas is None else quotas
    validate_config(window, stride, gap, quotas, role_order)
    reference_root, raw_root = Path(reference_root), Path(raw_root)
    inventory_path = reference_root / 'inventory.json'
    records_path = reference_root / 'raw_records.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    records = json.loads(records_path.read_text(encoding='utf-8'))
    if not inventory or not records:
        raise ValueError('Reference inventory and raw record list must be nonempty')
    known_ids = {task['task_id'] for task in inventory}
    if len(known_ids) != len(inventory):
        raise ValueError('Duplicate task IDs')
    if task_ids is not None:
        wanted = set(task_ids)
        if not wanted or wanted - known_ids:
            raise ValueError('Requested task ID absent from reference inventory')
        inventory = [task for task in inventory if task['task_id'] in wanted]
    classes = inventory[0]['classes']
    if not isinstance(classes, int) or classes < 2:
        raise ValueError('Transfer classification requires at least two classes')
    if any(task['dataset'] != dataset or task['classes'] != classes for task in inventory):
        raise ValueError('Reference tasks have inconsistent dataset or class count')
    by_record = {}
    seen_sha = set()
    for record in records:
        identity = (record['condition'], record['label'])
        if identity in by_record or not isinstance(record['label'], int) or record['label'] not in range(classes):
            raise ValueError('Require exactly one raw record per condition and class')
        if not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64:
            raise ValueError('Raw record SHA256 missing')
        if record['sha256'] in seen_sha:
            raise ValueError('A raw recording is assigned to more than one condition/class')
        relative = Path(record['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix.lower() != '.mat':
            raise ValueError('Raw recording path must be a relative MAT path below raw_root')
        if (not isinstance(record.get('shape'), list) or len(record['shape']) != 2
                or any(not isinstance(x, int) or x <= 0 for x in record['shape'])):
            raise ValueError('Raw recording requires a positive [time, channels] shape')
        seen_sha.add(record['sha256'])
        by_record[identity] = record
    required_banks = {}
    for task in inventory:
        for role in ROLES:
            side = task[role]
            identity = (side['condition'], side['group'], role)
            columns = side['raw_columns_zero_based']
            if (not isinstance(columns, list) or not columns or
                    any(not isinstance(x, int) or x < 0 for x in columns) or
                    len(set(columns)) != len(columns)):
                raise ValueError('Invalid raw sensor columns')
            for label in range(classes):
                if (side['condition'], label) not in by_record:
                    raise ValueError('Task condition lacks a raw class recording')
                if max(columns) >= by_record[(side['condition'], label)]['shape'][1]:
                    raise ValueError('Raw sensor column exceeds metadata channel count')
            if identity in required_banks and required_banks[identity] != columns:
                raise ValueError('One role bank has conflicting raw column definitions')
            required_banks[identity] = columns
    required_conditions = {condition for condition, _, _ in required_banks}
    layouts = {}
    minimum_capacity = {role: {part: None for part in PARTITIONS} for role in ROLES}
    for condition in sorted(required_conditions):
        if {label for cond, label in by_record if cond == condition} != set(range(classes)):
            raise ValueError(f'{condition} lacks one raw record for every class')
        for label in range(classes):
            record = by_record[(condition, label)]
            layout = plan_record(record, window, stride, gap, quotas, role_order)
            layouts[(condition, label)] = layout
            for role in ROLES:
                for part in PARTITIONS:
                    current = minimum_capacity[role][part]
                    value = layout['capacity'][role][part]
                    minimum_capacity[role][part] = value if current is None else min(current, value)
    task_origin_audit = {}
    for task in inventory:
        role_rows = {}
        for role in ROLES:
            condition = task[role]['condition']
            role_rows[role] = [row for label in range(classes) for part in PARTITIONS
                               for row in layouts[(condition, label)]['rows'][role][part]]
        task_origin_audit[task['task_id']] = validate_task_origins(
            role_rows['source'], role_rows['target'], window, stride, gap, quotas, classes)
    plan = {'schema_version': 3, 'protocol': 'antipaired_role_banks_v1',
            'loader_compatible': False, 'dataset': dataset, 'classes': classes,
            'task_ids': [task['task_id'] for task in inventory],
            'role_order': role_order, 'window_size': window, 'stride_samples': stride,
            'gap_samples': gap, 'partition_breaks': ['floor(55*L/100)', 'floor(85*L/100)'],
            'quotas_per_class': quotas, 'minimum_role_capacity_per_record': minimum_capacity,
            'task_origin_audit': task_origin_audit,
            'reference_sha256': {'inventory.json': file_sha256(inventory_path),
                                 'raw_records.json': file_sha256(records_path)},
            'role_banks': [{'condition': c, 'group': g, 'role': role,
                            'raw_columns_zero_based': columns}
                           for (c, g, role), columns in sorted(required_banks.items())],
            'no_existing_target_label_array_read': True,
            'target_final_policy': 'All target labels are written only in a sibling private audit directory; training loader must not decode them.'}
    if dataset == 'seu_gearbox':
        if {record.get('sampling_rate_hz') for record in records} != {5120}:
            raise ValueError('SEU intake requires explicit 5120 Hz metadata')
        plan['sampling_rate_hz'] = 5120
        plan['sampling_rate_evidence'] = 'User supplied 2026-09-15; not embedded in converted MAT'
        plan['channel_mapping_evidence'] = 'User supplied seven vibration columns; original torque removed according to supplied legend'
    plan['plan_sha256'] = json_sha256({k: v for k, v in plan.items() if k != 'plan_sha256'})
    return plan, inventory, by_record, layouts, required_banks


def _bank_path(output, condition, group, role):
    return output / 'domains' / f'{condition}_{group}_{role}'


def _public_row(row, role, token_salt):
    """Expose only a split index and an unjoinable opaque row token."""
    material = f'{token_salt}:{role}:{row["record_id"]}:{row["start"]}:{row["end"]}'
    return {'row_token': hashlib.sha256(material.encode('ascii')).hexdigest(),
            'partition': row['partition']}


def build(reference_root, raw_root, output, dataset, role_order='source_first',
          window=2048, gap=24000, stride=None, quotas=None, task_ids=None):
    """Write independent schema-3 arrays from raw MAT, never reference label NPYs."""
    plan, tasks, records, layouts, bank_specs = prepare(
        reference_root, raw_root, dataset, role_order, window, gap, stride, quotas, task_ids)
    output, raw_root = Path(output), Path(raw_root)
    script_path = Path(__file__).resolve()
    code_sha_before = file_sha256(script_path)
    # Check every raw input before creating any output. Avoid a partial version
    # whose first rows come from a changed or substituted recording.
    for (condition, label), record in sorted(records.items()):
        if condition in {item['condition'] for item in plan['role_banks']}:
            path = raw_root / record['relative_path']
            if not path.is_file() or file_sha256(path) != record['sha256']:
                raise ValueError(f'Raw SHA256 mismatch for {condition} class {label}')
    private = output.parent / f'{output.name}_private_audit'
    if output.exists() or private.exists():
        raise FileExistsError('Both public output and private audit paths must be new')
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    (output / 'code').mkdir()
    shutil.copy2(script_path, output / 'code' / script_path.name)
    (private / 'reference').mkdir()
    for name, expected in plan['reference_sha256'].items():
        source = Path(reference_root) / name
        if file_sha256(source) != expected:
            raise RuntimeError(f'Reference {name} changed during construction')
        shutil.copy2(source, private / 'reference' / name)
        if file_sha256(private / 'reference' / name) != expected:
            raise RuntimeError(f'Reference {name} snapshot differs from plan')
    token_salt = secrets.token_hex(32)
    shuffle_seed = secrets.randbits(128)
    rng = np.random.default_rng(shuffle_seed)
    plan['builder_code_sha256'] = code_sha_before
    # The prepared plan SHA is stable across dry-run and build; code is an
    # additional frozen execution identity rather than an unplanned mutation.
    save_json(output / 'build_plan.json', plan)
    banks = {}
    classes = plan['classes']
    for item in plan['role_banks']:
        condition, group, role = item['condition'], item['group'], item['role']
        directory = _bank_path(output, condition, group, role)
        directory.mkdir(parents=True)
        per_class = sum(plan['quotas_per_class'][role].values())
        array = np.lib.format.open_memmap(directory / 'data.npy', mode='w+', dtype=np.float32,
                                          shape=(classes * per_class, len(item['raw_columns_zero_based']), window))
        total = classes * per_class
        if role == 'source':
            permutation = np.arange(total)
        else:
            canonical_labels = np.repeat(np.arange(classes), per_class)
            for _ in range(64):
                permutation = rng.permutation(total)
                shuffled_labels = np.empty(total, dtype=np.int64)
                shuffled_labels[permutation] = canonical_labels
                if all(len(set(shuffled_labels[i:i + per_class])) > 1
                       for i in range(0, total, per_class)):
                    break
            else:
                raise RuntimeError('Failed to mix target classes across all bank blocks')
        banks[(condition, group, role)] = {
            'array': array, 'rows': [None] * total, 'labels': np.empty(total, dtype=np.int64),
            'next': 0, 'permutation': permutation,
            'columns': item['raw_columns_zero_based'], 'path': directory,
            'per_class': per_class}
    for condition, label in sorted(layouts):
        record = records[(condition, label)]
        raw_path = raw_root / record['relative_path']
        variables = {name: shape for name, shape, _ in whosmat(raw_path)}
        if variables.get('data') != tuple(record['shape']):
            raise ValueError('Raw MAT data shape changed')
        raw = loadmat(raw_path, variable_names=['data'])['data']
        if raw.shape != tuple(record['shape']) or raw.dtype.kind not in 'fiu':
            raise ValueError('Raw MAT data must be numeric [time,channels]')
        layout = layouts[(condition, label)]
        for (bank_condition, group, role), bank in banks.items():
            if bank_condition != condition:
                continue
            columns = bank['columns']
            if max(columns) >= raw.shape[1]:
                raise ValueError('Raw sensor column exceeds MAT data width')
            for part in PARTITIONS:
                for row in layout['rows'][role][part]:
                    sample = raw[row['start']:row['end'], columns].T.astype(np.float32)
                    if not np.isfinite(sample).all():
                        raise ValueError('Nonfinite selected raw signal')
                    destination = int(bank['permutation'][bank['next']])
                    bank['array'][destination] = sample
                    bank['rows'][destination] = dict(row)
                    bank['labels'][destination] = label
                    bank['next'] += 1
        del raw
    bank_files = {}
    private_target_files = {}
    private_bank_origins = {}
    for key, bank in banks.items():
        bank['array'].flush()
        del bank['array']
        condition, group, role = key
        directory = bank['path']
        if bank['next'] != len(bank['rows']) or any(row is None for row in bank['rows']):
            raise RuntimeError('Role bank row count differs from planned class quotas')
        public_rows = [_public_row(row, role, token_salt) for row in bank['rows']]
        save_json(directory / 'row_provenance.json', public_rows)
        bank['public_rows'] = public_rows
        visible_names = ['data.npy', 'row_provenance.json']
        if role == 'source':
            np.save(directory / 'label.npy', bank['labels'])
            visible_names.append('label.npy')
        else:
            label_dir = private / 'target_labels' / f'{condition}_{group}'
            label_dir.mkdir(parents=True)
            np.save(label_dir / 'label.npy', bank['labels'])
            private_target_files[key] = {'path': str(label_dir / 'label.npy'),
                                         'sha256': file_sha256(label_dir / 'label.npy')}
        bank_files[key] = {name: {'path': str(directory / name), 'sha256': file_sha256(directory / name)}
                           for name in visible_names}
        private_bank_origins[f'{condition}_{group}_{role}'] = {
            'rows': bank['rows'], 'canonical_to_bank_index': bank['permutation'].tolist()}
    task_hashes = {}
    for task in tasks:
        task_dir = output / 'tasks' / task['task_id']
        task_dir.mkdir(parents=True)
        source_key = (task['source']['condition'], task['source']['group'], 'source')
        target_key = (task['target']['condition'], task['target']['group'], 'target')
        source_rows, target_rows = banks[source_key]['rows'], banks[target_key]['rows']
        audit = validate_task_origins(source_rows, target_rows, window, plan['stride_samples'], gap,
                                      plan['quotas_per_class'], classes)
        files = {'source_data': bank_files[source_key]['data.npy'],
                 'source_label': bank_files[source_key]['label.npy'],
                 'target_data': bank_files[target_key]['data.npy']}
        manifest = {'schema_version': 3, 'protocol': 'antipaired_role_banks_v1',
                    'loader_compatible': False, 'task_id': task['task_id'], 'dataset': dataset,
                    'task_type': task['task_type'], 'source': task['source'], 'target': task['target'],
                    'window_size': window, 'stride_samples': plan['stride_samples'],
                    'gap_samples': gap, 'role_order': role_order, 'classes': classes,
                    'quotas_per_class': plan['quotas_per_class'], 'plan_sha256': plan['plan_sha256'],
                    'data_files': files,
                    'target_label_sha256_for_sealed_evaluation': private_target_files[target_key]['sha256'],
                    'row_provenance': {'source': banks[source_key]['public_rows'],
                                       'target': banks[target_key]['public_rows']},
                    'cross_role_audit': audit, 'splits': {}}
        for split_name, role, part in (('source_train', 'source', 'train'),
                                       ('source_val', 'source', 'dev'),
                                       ('target_train', 'target', 'train'),
                                       ('target_test', 'target', 'dev'),
                                       ('target_final_test', 'target', 'final')):
            rows = source_rows if role == 'source' else target_rows
            manifest['splits'][split_name] = {'domain': role,
                                              'indices': [i for i, row in enumerate(rows)
                                                          if row['partition'] == part]}
        manifest['reserved_source_final_indices'] = [i for i, row in enumerate(source_rows)
                                                      if row['partition'] == 'final']
        if dataset == 'seu_gearbox':
            manifest['sampling_rate_hz'] = plan['sampling_rate_hz']
            manifest['sampling_rate_evidence'] = plan['sampling_rate_evidence']
            manifest['channel_mapping_evidence'] = plan['channel_mapping_evidence']
        save_json(task_dir / 'task_manifest.json', manifest)
        task_hashes[task['task_id']] = file_sha256(task_dir / 'task_manifest.json')
    if file_sha256(script_path) != code_sha_before:
        raise RuntimeError('Builder source changed during construction')
    for name, expected in plan['reference_sha256'].items():
        if file_sha256(Path(reference_root) / name) != expected:
            raise RuntimeError(f'Reference {name} changed during construction')
    private_audit = {
        'schema_version': 3, 'plan_sha256': plan['plan_sha256'],
        'token_salt': token_salt, 'target_shuffle_seed': shuffle_seed,
        'raw_records': sorted(({'condition': r['condition'], 'label': r['label'],
                                'relative_path': r['relative_path'], 'sha256': r['sha256']}
                               for r in records.values()),
                              key=lambda r: (r['condition'], r['label'])),
        'bank_origins': private_bank_origins,
        'target_label_files': {f'{c}_{g}_{role}': value
                               for (c, g, role), value in private_target_files.items()},
        'task_origin_audit': plan['task_origin_audit']}
    save_json(private / 'private_audit.json', private_audit)
    result = {'schema_version': 3, 'protocol': 'antipaired_role_banks_v1',
              'plan_sha256': plan['plan_sha256'], 'builder_code_sha256': code_sha_before,
              'task_manifest_sha256': task_hashes,
              'bank_file_sha256': {f'{c}_{g}_{role}': value for (c, g, role), value in bank_files.items()},
              'private_audit_sha256': file_sha256(private / 'private_audit.json'),
              'record_independent': False, 'target_final_evaluated': False,
              'loader_compatible': False}
    save_json(output / 'version_manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=sorted(DEFAULTS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role-order', choices=('source_first', 'target_first'), required=True)
    parser.add_argument('--window', type=int, default=2048)
    parser.add_argument('--gap', type=int, default=24000)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--tasks', help='Comma-separated task IDs; default all')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true', help='Check metadata capacity only; write nothing')
    action.add_argument('--build', action='store_true', help='Read raw MAT and write an immutable schema-3 version')
    args = parser.parse_args(argv)
    selected = None if not args.tasks else [x.strip() for x in args.tasks.split(',') if x.strip()]
    if args.build:
        result = build(args.reference_root, args.raw_root, args.output, args.dataset,
                       args.role_order, args.window, args.gap, args.stride, task_ids=selected)
        print(json.dumps({'status': 'built_not_trainable_by_old_loader',
                          'plan_sha256': result['plan_sha256'], 'output': str(args.output.resolve())}))
    else:
        plan, *_ = prepare(args.reference_root, args.raw_root, args.dataset,
                           args.role_order, args.window, args.gap, args.stride, task_ids=selected)
        print(json.dumps({'status': 'capacity_checked_no_output', 'plan_sha256': plan['plan_sha256'],
                          'minimum_role_capacity_per_record': plan['minimum_role_capacity_per_record'],
                          'task_ids': plan['task_ids']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
