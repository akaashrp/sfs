"""Refit and audit SFS batch-latency coefficients from destination traces of a `refit` serving configuration."""
import argparse
import csv
from pathlib import Path

from scripts.cloud.common import read, write, digest
from scripts.cloud.serving.profiles import profile as load_profile, for_configuration

NAMES=('intercept','prefill_coeff','prefill_sq_coeff','decode_coeff','sum_coeff','sum_sq_coeff')
# The sixth coefficient is stored as sum_sq_coeff under either feature set -- the bundle and
# service_metrics_config.build_simulation_args convention -- and the declared feature set decides whether
# vLLM reads it as the sum of squared context lengths (legacy) or the prefill x processed-context cross term.
FEATURES={'p':'prefill_coeff','d':'decode_coeff','s':'sum_coeff','p_sq_sum':'prefill_sq_coeff','s_sq':'sum_sq_coeff','p_x_ctx':'sum_sq_coeff'}
FEATURE_SETS=('legacy','cross_term')
# A sanity floor on the regression, not the accuracy gate: the batch-residual audit replays the fitted
# coefficients against the engine's own batches during qualification and is what binds.
MINIMUM_R2=.93


def refit_profile(profile):
    if profile.coefficient_policy!='refit':
        raise ValueError(f'The {profile.name} profile retains the canonical SFS coefficients; nothing to fit or load')
    return profile


def traces(folder):
    found={p.name[len('calibration_trace_'):-4]:p for p in sorted(Path(folder).glob('calibration_trace_*.csv'))}
    if not found:raise ValueError(f'No calibration traces under {folder}')
    return found


def predict(frame, coefficients, feature_set='legacy'):
    import numpy as np
    from sfs_core.regression.two_part_fit import build_feature_matrix
    x,names=build_feature_matrix(frame,feature_set=feature_set)
    return x@np.array([coefficients[FEATURES[n]] for n in names])+coefficients['intercept']


def diagnostics(frame, predicted):
    observed=frame['exec'].to_numpy();prefill=frame['prefill'].to_numpy();decode=frame['decode'].to_numpy();nonempty=(prefill+decode)>0
    total=float(((observed-observed.mean())**2).sum());residual=float(((observed-predicted)**2).sum())
    groups={'all':nonempty|True,'pure_decode':(prefill==0)&nonempty,'prefill':prefill>0,'mixed_prefill_decode':(prefill>0)&(decode>0)}
    return {'rows':int(len(observed)),'r2_all_rows':1.-residual/total if total>0 else 1.,'mae_s_all_rows':float(abs(observed-predicted).mean()),
        'negative_nonempty_rows':int((predicted[nonempty]<0).sum()),
        'groups':{g:{'rows':int(m.sum()),'mae_ms':float(abs(observed[m]-predicted[m]).mean()*1e3),'mean_bias_ms':float((predicted[m]-observed[m]).mean()*1e3),
                     'mean_observed_ms':float(observed[m].mean()*1e3)} for g,m in groups.items() if m.any()}}


def fit(calibration, output, profile):
    """Same estimator as derive_nonnegative_calibration: Huber inliers, then NNLS."""
    import pandas as pd
    from sfs_core.regression.two_part_fit import fit_two_part_from_df
    profile=refit_profile(profile);models={};feature_set=profile.fit_feature_set
    for model,trace in traces(calibration).items():
        frame=pd.read_csv(trace)
        result=fit_two_part_from_df(frame,stall_percentile=99.9,feature_set=feature_set,nonnegative_coefficients=True)
        c={'intercept':float(result.base_model.intercept_),**{FEATURES[n]:float(v) for n,v in zip(result.feature_names,result.base_model.coef_)}}
        d=diagnostics(frame,predict(frame,c,feature_set))
        if d['r2_all_rows']<MINIMUM_R2:raise ValueError(f"{model} {profile.configuration_id} batch fit R^2={d['r2_all_rows']:.4f} is below {MINIMUM_R2}")
        models[model]={**c,'feature_set':feature_set,'coefficient_constraint':result.coefficient_constraint,'fit_rows':int(len(frame)),
            'fit_inlier_rows':int(result.inlier_mask.sum()),'stall_probability':result.stall_probability,'fit_prediction_diagnostics':d,
            'trace':str(trace),'trace_sha256':digest(trace)}
    write(output,{'schema_version':1,'configuration_id':profile.configuration_id,'profile':profile.settings,'status':'FITTED_FOR_CONFIGURATION_REVIEW_REQUIRED',
        'method':f'Huber inliers at the 99.9th residual percentile, then nonnegative least squares on {feature_set} features; destination calibration traces of this configuration only',
        'models':models})


def load(path, profile):
    payload=read(path);profile=refit_profile(profile)
    if payload.get('configuration_id')!=profile.configuration_id or payload.get('profile')!=profile.settings:
        raise ValueError(f'Coefficients were not fitted for the {profile.configuration_id} configuration')
    coefficients={}
    for model,row in payload['models'].items():
        if any(float(row[k])<0 for k in NAMES) or row['fit_prediction_diagnostics']['r2_all_rows']<MINIMUM_R2:
            raise ValueError(f'Rejected {profile.configuration_id} coefficients for {model}')
        # The feature set travels with the values: dropping it is how the Ministral cross-term fit reached
        # its engines as legacy. A file fitted under another feature set than the profile declares is refused.
        feature_set=row.get('feature_set','legacy')
        if feature_set not in FEATURE_SETS or feature_set!=profile.fit_feature_set:
            raise ValueError(f'{profile.configuration_id} expects {profile.fit_feature_set} coefficients; {model} was fitted as {feature_set}')
        coefficients[model]={**{k:float(row[k]) for k in NAMES},'feature_set':feature_set}
    return coefficients


def validate(coefficients_path, qualification, output, profile):
    """Predicted-versus-observed batch times on batches recorded after calibration ended."""
    import pandas as pd
    fitted=load(coefficients_path,profile);models={}
    for model,trace in traces(qualification).items():
        with trace.open() as f:end=max(float(r['ts']) for r in csv.DictReader(f))
        frame=pd.read_csv(Path(qualification)/f'batch_stats_{model}.csv');frame=frame[(frame['ts']>end)&(frame['exec']>0)].reset_index(drop=True)
        if not len(frame):raise ValueError(f'No independent post-calibration batches for {model}')
        fs=fitted[model]['feature_set']
        models[model]={'calibration_last_epoch':end,'feature_set':fs,'independent':diagnostics(frame,predict(frame,fitted[model],fs)),
                       'calibration':diagnostics(pd.read_csv(trace),predict(pd.read_csv(trace),fitted[model],fs))}
    write(output,{'configuration_id':profile.configuration_id,'coefficients_sha256':digest(coefficients_path),'qualification':str(qualification),
        'scope':'Fitted SFS coefficients of this configuration on independently generated post-calibration smoke and load-probe batches; no refitting','models':models})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','validate'])
    p.add_argument('--profile',required=True,help='refit serving profile (fcfs, chunk8192, kv_constrained)')
    p.add_argument('--calibration',type=Path,help='worker output directory holding calibration_trace_<model>.csv')
    p.add_argument('--coefficients',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='fit':fit(a.calibration,a.output,load_profile(a.profile))
    else:validate(a.coefficients,a.calibration,a.output,load_profile(a.profile))
    print(a.output)
