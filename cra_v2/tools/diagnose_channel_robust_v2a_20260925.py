"""Artifact-only mechanism diagnosis; no dataset array or target-label access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/mnt/d/3800')
EXP = ROOT / 'experiments/20260925_channel_robust_v2a_screen_v5'
torch.set_num_threads(1)

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def epochs(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]

def mean(rows,key):
    return float(np.mean([r[key] for r in rows]))

def displacement(a,b,keys):
    x = torch.cat([a[k].reshape(-1).double() for k in keys])
    y = torch.cat([b[k].reshape(-1).double() for k in keys])
    return {'l2':float((x-y).norm()),'relative_l2':float((x-y).norm()/x.norm().clamp_min(1e-20))}

def main():
    dev = read(EXP/'dev_results.json')
    summary = read(EXP/'summary.json')
    output = {'experiment':str(EXP),'plan_sha256':sha(EXP/'plan.json'),
              'dev_results_sha256':sha(EXP/'dev_results.json'),
              'existing_artifacts_only':True,'new_training':False,
              'target_train_labels_opened_by_diagnostic':False,
              'target_dev_labels_opened_by_diagnostic':False,
              'target_final_opened_by_diagnostic':False,'pairs':[],'anchors':[]}
    for task in ('WP-S0','WP-D1','PG-S1','PG-D1'):
        anchor_path = EXP/'runs'/f'{task}_source012_common_anchor_seed42_cuda'
        a = torch.load(anchor_path/'best_model.pth',map_location='cpu',weights_only=True)
        ae = epochs(anchor_path/'epoch_metrics.jsonl')
        output['anchors'].append({'task':task,'selected_epoch':a['selected_epoch'],
                                  'source_val_accuracy':a['source_val']['accuracy'],
                                  'source_val_ce':a['source_val']['loss'],
                                  'kernel':a['kernel_contract']})
        for slots in ([0,1,2],[0],[1],[2]):
            label = 'all3' if len(slots)==3 else f'slot{slots[0]}'
            ck = ['common_anchor_control','channel_robust_alignment_v2a']
            dirs = [EXP/'runs'/f'{task}_source012_target_{label}_{v}_seed42_cuda' for v in ck]
            checkpoints = [torch.load(p/'best_model.pth',map_location='cpu',weights_only=True) for p in dirs]
            log = [epochs(p/'epoch_metrics.jsonl') for p in dirs]
            results = [next(r for r in dev if r['arm']['task']==task and r['arm']['target']==slots
                            and r['arm']['variant']==v) for v in ck]
            control,candidate = results
            z0,z1 = [z['model_state_dict'] for z in checkpoints]
            candidate_e = log[1]
            last = candidate_e[-1]
            row = {'task':task,'target':slots,'control_accuracy':control['accuracy'],
                   'candidate_accuracy':candidate['accuracy'],
                   'accuracy_delta':candidate['accuracy']-control['accuracy'],
                   'macro_f1_delta':candidate['macro_f1']-control['macro_f1'],
                   'control_recall':control['per_class_recall'],'candidate_recall':candidate['per_class_recall'],
                   'control_confusion':control['confusion_matrix'], 'candidate_confusion':candidate['confusion_matrix'],
                   'new_zero_recall_classes':[i for i,(x,y) in enumerate(zip(control['per_class_recall'],candidate['per_class_recall'])) if x>0 and y==0],
                   'alignment_steps':last['alignment_steps_total'],
                   'applied_steps':last['applied_alignment_steps_total'],
                   'applied_fraction':last['applied_alignment_steps_total']/last['alignment_steps_total'],
                   'mean_cosine':mean(candidate_e,'mean_alignment_source_cosine'),
                   'mean_fused_mmd':mean(candidate_e,'fused_mmd'),
                   'mean_channel_mmd':mean(candidate_e,'channel_mmd'),
                   'mean_source_total':mean(candidate_e,'total_source'),
                   'weighted_alignment_scalar_over_source_scalar':.02*mean(candidate_e,'alignment')/mean(candidate_e,'total_source'),
                   'fused_negative_fraction':float(np.mean([e['signed_mmd']['fused']['negative_fraction'] for e in candidate_e])),
                   'channel_negative_fraction':float(np.mean([e['signed_mmd']['channel']['negative_fraction'] for e in candidate_e])),
                   'control_source_val_accuracy_min':min(e['source_val']['accuracy'] for e in log[0]),
                   'candidate_source_val_accuracy_min':min(e['source_val']['accuracy'] for e in log[1]),
                   'control_source_val_ce_last':log[0][-1]['source_val']['loss'],
                   'candidate_source_val_ce_last':log[1][-1]['source_val']['loss'],
                   'source_buffers_equal_anchor':all(torch.equal(z[k],a['model_state_dict'][k]) for z in (z0,z1) for k in ('source_center','source_scale')),
                   'paired_target_buffers_exact':all(torch.equal(z0[k],z1[k]) for k in ('target_center','target_scale','target_calibrated')),
                   'paired_source_rows_exact':checkpoints[0]['branch_source_indices_sha256']==checkpoints[1]['branch_source_indices_sha256'],
                   'paired_source_masks_exact':checkpoints[0]['branch_source_masks_sha256']==checkpoints[1]['branch_source_masks_sha256'],
                   'paired_initial_state_exact':checkpoints[0]['branch_start_identity']==checkpoints[1]['branch_start_identity'],
                   'checkpoint_sha256':[sha(p/'best_model.pth') for p in dirs],
                   'epochs_sha256':[sha(p/'epoch_metrics.jsonl') for p in dirs]}
            for layer in ('encoder.','classifier.','fusion_score'):
                row[layer+'displacement'] = displacement(z0,z1,[k for k in z0 if k.startswith(layer)])
            for name, rr in zip(('control','candidate'), results):
                cm=np.asarray(rr['confusion_matrix'])
                share=cm.sum(0)/cm.sum()
                row[name+'_prediction_share']=share.tolist()
                row[name+'_prediction_entropy_normalized']=float(-np.sum(share[share>0]*np.log(share[share>0]))/np.log(len(share)))
            row['prediction_share_tv']=float(.5*np.abs(np.array(row['control_prediction_share'])-row['candidate_prediction_share']).sum())
            for suffix, es in [('first5',candidate_e[:5]),('last5',candidate_e[-5:])]:
                row[suffix]={k:mean(es,k) for k in ('alignment_applied_fraction','fused_mmd','channel_mmd','total_source','source_ce')}
            output['pairs'].append(row)
    ps=output['pairs']
    output['aggregate']={'mean_accuracy_delta':mean(ps,'accuracy_delta'),'mean_macro_f1_delta':mean(ps,'macro_f1_delta'),
       'applied_fraction_weighted':sum(r['applied_steps'] for r in ps)/sum(r['alignment_steps'] for r in ps),
       'applied_fraction_range':[min(r['applied_fraction'] for r in ps),max(r['applied_fraction'] for r in ps)],
       'mean_cosine':mean(ps,'mean_cosine'),
       'source_buffers_equal_anchor':all(r['source_buffers_equal_anchor'] for r in ps),
       'paired_target_buffers_exact':all(r['paired_target_buffers_exact'] for r in ps),
       'pair_start_rows_masks_exact':all(r['paired_initial_state_exact'] and r['paired_source_rows_exact'] and r['paired_source_masks_exact'] for r in ps),
       'new_zero_recall_cases':sum(len(r['new_zero_recall_classes']) for r in ps),
       'all_source_val_accuracy_one':all(r['control_source_val_accuracy_min']==r['candidate_source_val_accuracy_min']==1 for r in ps),
       'joint_improved':sum(r['accuracy_delta']>0 and r['macro_f1_delta']>0 for r in ps)}
    (ROOT/'audit/channel_robust_v2a_mechanism_diagnostic_20260925.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(output['aggregate'],indent=2))
    for r in ps:
        print(r['task'],r['target'], 'dacc/f1',round(100*r['accuracy_delta'],3),round(100*r['macro_f1_delta'],3),
              'applied',round(100*r['applied_fraction'],2),'cos',round(r['mean_cosine'],3),
              'MMD',round(r['mean_fused_mmd'],4),round(r['mean_channel_mmd'],4),
              'neg',round(r['fused_negative_fraction'],3),round(r['channel_negative_fraction'],3),
              'scalar ratio',round(r['weighted_alignment_scalar_over_source_scalar'],2),
              'newzero',r['new_zero_recall_classes'])

if __name__=='__main__':main()
