"""Training-independent protocol checks; importing this module never imports torch."""

import math


def validate_synchronized_pairing(manifest):
    """Prove source/target train rows describe the same acquisition interval."""
    if not isinstance(manifest, dict) or manifest.get('protocol') != 'synchronized_time_block_split':
        raise ValueError('Paired adaptation requires the audited synchronized time-block manifest')
    provenance = manifest.get('row_provenance', {})
    splits = manifest.get('splits', {})
    source_rows, target_rows = provenance.get('source'), provenance.get('target')
    source_indices = splits.get('source_train', {}).get('indices')
    target_indices = splits.get('target_train', {}).get('indices')
    if not all(isinstance(value, list) for value in (source_rows, target_rows, source_indices, target_indices)):
        raise ValueError('Paired adaptation requires explicit row provenance and training indices')
    if source_indices != target_indices or not source_indices:
        raise ValueError('Paired source and target training indices must be identical and nonempty')
    fields = ('record_id', 'start', 'end', 'partition')
    for index in source_indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < min(len(source_rows), len(target_rows)):
            raise ValueError('Invalid synchronized training row index')
        if any(source_rows[index].get(field) != target_rows[index].get(field) for field in fields):
            raise ValueError('Source and target training windows are not the same acquisition interval')
        if source_rows[index].get('partition') != 'train':
            raise ValueError('Paired adaptation may use training-partition windows only')
    return {'paired_training_rows': len(source_indices), 'identity_fields': list(fields),
            'target_labels_used': False, 'validation_or_final_rows_used': False}


def resolve_seeds(args):
    if args.seed_list.strip():
        seeds = [int(value.strip()) for value in args.seed_list.split(',') if value.strip()]
    else:
        if args.repeat_times <= 0:
            raise ValueError('repeat_times must be positive')
        seeds = [args.seed + offset for offset in range(args.repeat_times)]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Independent repeats require a non-empty list of distinct seeds')
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError('Seeds must be in [0, 2**32)')
    return seeds


def validate_protocol(args):
    """Fail before allocating a device or loading any experimental data."""
    if args.max_epoch <= 0 or args.batch_size <= 0 or args.print_step <= 0:
        raise ValueError('max_epoch, batch_size and print_step must be positive')
    if not 0 < args.target_test_size < 1:
        raise ValueError('target_test_size must be between 0 and 1')
    source_test = getattr(args, 'source_test_size', 0.0)
    source_val = getattr(args, 'source_val_size', 0.2)
    final_target = getattr(args, 'target_final_test_size', 0.0)
    if not 0 <= source_test < 1 or not 0 < source_val < 1 or source_test + source_val >= 1:
        raise ValueError('Source train/validation/test fractions must leave non-empty splits')
    if not 0 <= final_target < 1 or final_target + args.target_test_size >= 1:
        raise ValueError('Target adaptation/validation/final fractions must leave non-empty splits')
    if getattr(args, 'predefined_split_path', ''):
        if args.experiment_mode != 'uda' or not args.strict_uda or source_test != 0:
            raise ValueError('Predefined raw-time splits require strict UDA and source_test_size=0; both final regions stay sealed')
    if not 0 <= getattr(args, 'selection_start_epoch', 0) < args.max_epoch:
        raise ValueError('selection_start_epoch must precede max_epoch')
    if not 0 < getattr(args, 'gpu_memory_fraction', 0.4) <= 1 or getattr(args, 'cpu_threads', 2) < 1:
        raise ValueError('Invalid GPU memory fraction or CPU thread count')
    pause_ms = getattr(args, 'step_pause_ms', 0.0)
    if isinstance(pause_ms, bool) or not isinstance(pause_ms, (int, float)) or not math.isfinite(pause_ms) or pause_ms < 0:
        raise ValueError('step_pause_ms must be finite and non-negative')
    if args.middle_epoch < 0 or args.num_workers < 0:
        raise ValueError('middle_epoch and num_workers must be non-negative')
    if getattr(args, 'model_kind', 'current_ticnn') in {'simple_cnn', 'channel_shared'}:
        if args.dual_branch_mode != 'shared' or args.adapter_mode != 'none' or not args.no_local_attention or args.transfer_channel_attention or getattr(args, 'norm_layer', 'group') != 'group':
            raise ValueError('Simple CNN requires shared, adapter_mode=none, no_local_attention and no transfer_channel_attention')
    if getattr(args, 'model_kind', '') == 'widan_reference':
        if args.norm_layer != 'batch' or args.adapter_mode != 'none' or not args.no_local_attention or args.transfer_channel_attention or args.dual_branch_mode != 'shared' or args.branch_reg_weight != 0:
            raise ValueError('WIDAN reference uses its own explicit branch setting, BatchNorm, no adapters/attention/branch regularizer')
        if args.adaptation_mode not in {'source_only', 'mmd'} or args.target_im_weight != 0:
            raise ValueError('Reference reconstruction permits only Source Only or explicit MMD; no extra modules')
        for name in ('reference_source_frequency', 'reference_target_frequency'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError('Reference domain frequency must be finite and positive')
        if not math.isfinite(args.reference_eps):
            raise ValueError('Reference eps must be finite')
    if getattr(args, 'model_kind', '') == 'spectral_shared':
        if args.experiment_mode != 'uda' or args.adaptation_mode not in {'source_only', 'mmd', 'mean', 'zscore', 'paired', 'dann'}:
            raise ValueError('Spectral candidate supports UDA Source Only, MMD, DANN, paired or target-statistics adaptation')
        if args.normlizetype != 'per-channel-mean-std' or args.norm_layer != 'group' or args.dual_branch_mode != 'shared' or args.branch_reg_weight != 0:
            raise ValueError('Spectral candidate requires per-channel-mean-std and neutral shared/group/zero branch options; it has no internal GroupNorm')
        if args.adapter_mode != 'none' or not args.no_local_attention or args.transfer_channel_attention or args.target_im_weight != 0:
            raise ValueError('Spectral candidate forbids unused adapters, attention and extra target objectives')
        statistics = getattr(args, 'spectral_target_stats', 'auto')
        if statistics not in {'auto', 'source', 'mean', 'zscore'}:
            raise ValueError('Unknown spectral target statistics mode')
        if args.adaptation_mode in {'mean', 'zscore'} and statistics not in {'auto', args.adaptation_mode}:
            raise ValueError('Legacy statistics adaptation mode conflicts with explicit spectral target statistics')
        if args.adaptation_mode == 'source_only' and statistics in {'mean', 'zscore'}:
            raise ValueError('Source Only cannot fit target statistics; use mean or zscore adaptation mode')
        if args.adaptation_mode in {'mean', 'zscore'} and args.eval_target_branch == 'source':
            raise ValueError('Statistics adaptation must use calibrated target statistics for target evaluation')
        if args.adaptation_mode == 'dann':
            if not args.strict_uda or args.model_selection != 'source_val' or args.debug_target_each_epoch:
                raise ValueError('DANN requires strict UDA and source-val-only checkpoint selection')
            if not args.predefined_split_path and final_target <= 0:
                raise ValueError('DANN requires a reserved target final holdout')
            if args.spectral_encoder != 'mlp' or statistics not in {'auto', 'source', 'zscore'}:
                raise ValueError('DANN requires the shared 64-dimensional MLP and source or target z-score statistics')
            if args.spectral_channel_subsets:
                raise ValueError('The minimal DANN baseline keeps source channel subsets disabled')
            if args.eval_target_branch == 'source':
                raise ValueError('DANN target evaluation must use the target spectral path')
            if (args.trade_off_distance != 'Step' or not math.isfinite(args.adaptation_weight)
                    or not 0 < args.adaptation_weight <= 1 or args.middle_epoch >= args.max_epoch - 1):
                raise ValueError('DANN requires a positive scheduled GRL strength and at least one effective epoch')
    elif (args.adaptation_mode in {'mean', 'zscore', 'paired'} or
          args.adaptation_mode == 'dann' or
          getattr(args, 'spectral_channel_subsets', False) or
          getattr(args, 'spectral_target_stats', 'auto') != 'auto'):
        raise ValueError('Statistics/paired adaptation and spectral subsets require --model_kind spectral_shared')
    if args.experiment_mode == 'uda' and args.strict_uda:
        if args.model_selection not in {'source_val', 'fixed_epoch'}:
            raise ValueError('Strict UDA forbids target labels for model selection; use source_val or fixed_epoch')
        if args.debug_target_each_epoch:
            raise ValueError('Strict UDA evaluates target_test only after the checkpoint is locked')
    if args.experiment_mode == 'uda' and args.adaptation_mode == 'source_only':
        if args.transfer_channel_attention or args.target_im_weight != 0:
            raise ValueError('Source Only forbids target-derived attention and target information losses')
        if args.eval_target_branch == 'target':
            raise ValueError('Source Only must evaluate the trained source branch')
    if args.experiment_mode == 'target_supervised' and args.model_selection == 'source_val':
        raise ValueError('Target-supervised diagnostics need a fixed epoch or a separately defined target validation set')
    resolve_seeds(args)


def protocol_name(args):
    if args.experiment_mode != 'uda':
        return args.experiment_mode + '_diagnostic'
    if args.model_selection in {'target_test', 'target_debug'} or args.debug_target_each_epoch:
        return 'target_label_debug_not_strict_uda'
    if not args.strict_uda:
        return 'uda_target_labels_exposed_not_strict'
    if args.adaptation_mode == 'paired':
        if getattr(args, 'predefined_split_path', ''):
            return 'paired_cross_view_synchronized_time_blocks_sealed_final_record_independence_unverified'
        return 'paired_cross_view_correspondence_not_unpaired_uda'
    if getattr(args, 'predefined_split_path', ''):
        return 'exploratory_uda_synchronized_time_blocks_sealed_final_record_independence_unverified'
    if getattr(args, 'target_final_test_size', 0.0) > 0:
        return 'exploratory_uda_sealed_final_holdout_record_independence_unverified'
    # Label isolation alone does not establish record-independent splits.
    return 'uda_labels_hidden_legacy_window_split_unverified'


def classification_metrics(confusion):
    """Rows are true classes, columns are predictions; zero division gives zero."""
    n = len(confusion)
    if not n or any(len(row) != n for row in confusion):
        raise ValueError('Confusion matrix must be non-empty and square')
    if any(value < 0 for row in confusion for value in row):
        raise ValueError('Confusion counts must be non-negative')
    support = [sum(row) for row in confusion]
    predicted = [sum(confusion[row][col] for row in range(n)) for col in range(n)]
    precision = [confusion[i][i] / predicted[i] if predicted[i] else 0.0 for i in range(n)]
    recall = [confusion[i][i] / support[i] if support[i] else 0.0 for i in range(n)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    total = sum(support)
    return {
        'accuracy': sum(confusion[i][i] for i in range(n)) / total if total else 0.0,
        'macro_f1': sum(f1) / n,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'support': support,
        'confusion_matrix': confusion,
    }
