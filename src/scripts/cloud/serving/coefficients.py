"""Refit and audit SFS batch-latency coefficients from destination traces of a `refit` serving configuration."""
import argparse
import csv
from pathlib import Path

from scripts.cloud.common import read, write, digest
from scripts.cloud.serving.profiles import profile as load_profile, for_configuration

NAMES=('intercept','prefill_coeff','prefill_sq_coeff','decode_coeff','sum_coeff','sum_sq_coeff')
FEATURES={'p':'prefill_coeff','d':'decode_coeff','s':'sum_coeff','p_sq_sum':'prefill_sq_coeff','s_sq':'sum_sq_coeff'}
# A sanity floor on the regression, not the accuracy gate: the batch-residual audit replays the fitted
# coefficients against the engine's own batches during qualification and is what binds. R^2 punishes a
# narrow load range rather than a poor fit -- the constrained configuration's 128-sequence cap truncates
# the batch-size range, so its qwen3-0.6b fit scored R^2 0.9446 while predicting median 1.024x of actual
# (p90 1.101, MAE 1.0 ms on 15 ms batches), better than the accepted canonical 0.6B fit at 1.178x.
MINIMUM_R2=.93


def refit_profile(profile):
    if profile.coefficient_policy!='refit':
        raise ValueError(f'The {profile.name} profile retains the canonical SFS coefficients; nothing to fit or load')
    return profile


def traces(folder):
    found={p.name[len('calibration_trace_'):-4]:p for p in sorted(Path(folder).glob('calibration_trace_*.csv'))}
    if not found:raise ValueError(f'No calibration traces under {folder}')
    return found


def predict(frame, coefficients):
    import numpy as np
    from sfs_core.regression.two_part_fit import build_feature_matrix
    x,names=build_feature_matrix(frame,feature_set='legacy')
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
    profile=refit_profile(profile);models={}
    for model,trace in traces(calibration).items():
        frame=pd.read_csv(trace)
        result=fit_two_part_from_df(frame,stall_percentile=99.9,feature_set='legacy',nonnegative_coefficients=True)
        c={'intercept':float(result.base_model.intercept_),**{FEATURES[n]:float(v) for n,v in zip(result.feature_names,result.base_model.coef_)}}
        d=diagnostics(frame,predict(frame,c))
        if d['r2_all_rows']<MINIMUM_R2:raise ValueError(f"{model} {profile.configuration_id} batch fit R^2={d['r2_all_rows']:.4f} is below {MINIMUM_R2}")
        models[model]={**c,'feature_set':'legacy','coefficient_constraint':result.coefficient_constraint,'fit_rows':int(len(frame)),
            'fit_inlier_rows':int(result.inlier_mask.sum()),'stall_probability':result.stall_probability,'fit_prediction_diagnostics':d,
            'trace':str(trace),'trace_sha256':digest(trace)}
    write(output,{'schema_version':1,'configuration_id':profile.configuration_id,'profile':profile.settings,'status':'FITTED_FOR_CONFIGURATION_REVIEW_REQUIRED',
        'method':'Huber inliers at the 99.9th residual percentile, then nonnegative least squares on legacy features; destination calibration traces of this configuration only',
        'models':models})


def load(path, profile):
    payload=read(path);profile=refit_profile(profile)
    if payload.get('configuration_id')!=profile.configuration_id or payload.get('profile')!=profile.settings:
        raise ValueError(f'Coefficients were not fitted for the {profile.configuration_id} configuration')
    coefficients={}
    for model,row in payload['models'].items():
        if any(float(row[k])<0 for k in NAMES) or row['fit_prediction_diagnostics']['r2_all_rows']<MINIMUM_R2:
            raise ValueError(f'Rejected {profile.configuration_id} coefficients for {model}')
        coefficients[model]={k:float(row[k]) for k in NAMES}
    return coefficients


def validate(coefficients_path, qualification, output, profile):
    """Predicted-versus-observed batch times on batches recorded after calibration ended."""
    import pandas as pd
    fitted=load(coefficients_path,profile);models={}
    for model,trace in traces(qualification).items():
        with trace.open() as f:end=max(float(r['ts']) for r in csv.DictReader(f))
        frame=pd.read_csv(Path(qualification)/f'batch_stats_{model}.csv');frame=frame[(frame['ts']>end)&(frame['exec']>0)].reset_index(drop=True)
        if not len(frame):raise ValueError(f'No independent post-calibration batches for {model}')
        models[model]={'calibration_last_epoch':end,'independent':diagnostics(frame,predict(frame,fitted[model])),
                       'calibration':diagnostics(pd.read_csv(trace),predict(pd.read_csv(trace),fitted[model]))}
    write(output,{'configuration_id':profile.configuration_id,'coefficients_sha256':digest(coefficients_path),'qualification':str(qualification),
        'scope':'Fitted SFS coefficients of this configuration on independently generated post-calibration smoke and load-probe batches; no refitting','models':models})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','validate'])
    p.add_argument('--profile',required=True,help='refit serving profile (fcfs, chunk8192)')
    p.add_argument('--calibration',type=Path,help='worker output directory holding calibration_trace_<model>.csv')
    p.add_argument('--coefficients',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='fit':fit(a.calibration,a.output,load_profile(a.profile))
    else:validate(a.coefficients,a.calibration,a.output,load_profile(a.profile))
    print(a.output)
