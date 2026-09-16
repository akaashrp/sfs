"""Audit final fitted timing heads on post-calibration smoke batches only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from scripts.cloud.common import digest,write
from scripts.prep.fit_methodology_calibration import _error_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    calibration=a.qualification/'timing_models/methodology_calibration.json'
    m=json.loads(calibration.read_text());results={}
    for model,artifact in m['models'].items():
        with (a.qualification/f'calibration_trace_{model}.csv').open() as f:
            end=max(float(r['ts']) for r in csv.DictReader(f))
        with (a.qualification/f'batch_stats_{model}.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if float(r['ts'])>end and float(r['exec'])>0 and float(r['decode'])>0]
        if not rows:raise ValueError('No independent post-calibration smoke decode rows')
        head=xgb.XGBRegressor(n_jobs=1);head.load_model(calibration.parent/artifact['tpot']['model_file'])
        x=np.array([[float(r[k]) for k in ('decode','prefill','sum_tokens')] for r in rows]);y=np.array([float(r['exec'])*1000 for r in rows]);pred=head.predict(x)
        groups={'all':np.ones(len(rows),dtype=bool),'pure_decode':x[:,1]==0,'mixed_prefill_decode':x[:,1]>0,'decode_at_most_8':x[:,0]<=8,'decode_above_8':x[:,0]>8}
        results[model]={'calibration_last_epoch':end,'final_head_sha256':digest(calibration.parent/artifact['tpot']['model_file']),
            'independent_smoke_rows':len(rows),'groups':{g:_error_report(y[mask],pred[mask]) for g,mask in groups.items() if mask.any()}}
    write(a.output,{'calibration_sha256':digest(calibration),'scope':'Final fitted TPOT heads on independently generated post-calibration smoke batches; no refitting on smoke','models':results})
    print(json.dumps(results))

if __name__=='__main__':main()
