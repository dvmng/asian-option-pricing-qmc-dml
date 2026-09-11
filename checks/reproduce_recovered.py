"""Run recovered historical algorithms with current paths, writing only reproduced outputs."""
from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT/'archive/recovered_scripts'
OUT=ROOT/'results/reproduced/recovered'
OUT.mkdir(parents=True,exist_ok=True)

def compare(a,b):
    x=pd.read_csv(a);y=pd.read_csv(b)
    assert x.shape==y.shape,(a,x.shape,y.shape)
    assert list(x.columns)==list(y.columns),a
    maximum=0.
    for col in x:
        if pd.api.types.is_numeric_dtype(x[col]):
            np.testing.assert_allclose(x[col],y[col],rtol=1e-8,atol=1e-10,equal_nan=True)
            d=(x[col]-y[col]).abs().max()
            if pd.notna(d):maximum=max(maximum,float(d))
        else:
            assert x[col].fillna('').equals(y[col].fillna('')),(a,col)
    return {'file':str(a.relative_to(ROOT)),'rows':len(x),'max_absolute_numeric_difference':maximum}

def run(script,args):
    with (OUT/(script+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-B',str(ARCH/script),*map(str,args)],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)

spec=importlib.util.spec_from_file_location('bootstrap',ARCH/'11_bootstrap_logou_design_bands.py')
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot)
prices=pd.read_csv(ROOT/'data/raw/global/ornn_h100.csv').price_avg.to_numpy()
draws,meta=boot.residual_bootstrap(prices,1/365,20000,20260826)
assert len(draws)==19994 and meta['n_nonstationary_discarded']==6
draws.to_csv(OUT/'bootstrap_draws.csv',index=False)
boot.percentile_table(draws).to_csv(OUT/'bootstrap_band_sensitivity.csv',index=False)
checks=[compare(OUT/name,ROOT/'results/calibration/bootstrap_bands'/name) for name in ['bootstrap_draws.csv','bootstrap_band_sensitivity.csv']]
run('05_phase2_h100_model_evidence.py',['--input',ROOT/'data/raw/global/ornn_h100.csv','--output-dir',OUT/'model_evidence'])
checks.append(compare(OUT/'model_evidence/expanding_one_step.csv',ROOT/'results/calibration/model_evidence/expanding_one_step.csv'))
actual=json.loads((OUT/'model_evidence/phase2_model_evidence.json').read_text())
expected=json.loads((ROOT/'results/calibration/model_evidence/phase2_model_evidence.json').read_text())
for key in ['hac_test_sq_loss_diff_rw_minus_ar1','hac_test_abs_loss_diff_rw_minus_ar1']:
    np.testing.assert_allclose(list(actual['one_step_expanding_cv'][key].values()),list(expected['one_step_expanding_cv'][key].values()),rtol=1e-8,atol=1e-10)
run('19_diagnose_compute_extreme_errors.py',['--data-dir',ROOT/'data/frozen/pricing_dataset','--evaluation-dir',ROOT/'results/ml/final_evaluation','--output-dir',OUT/'extreme_errors'])
run('20_diagnose_fixing_boundary_errors.py',['--phase4d-dir',OUT/'extreme_errors','--output-dir',OUT/'fixing_boundary'])
for name in ['top50_fixing_boundary_summary.csv','error_by_fixing_boundary_distance.csv','error_by_n_fix_future.csv']:
    checks.append(compare(OUT/'fixing_boundary'/name,ROOT/'results/diagnostics/fixing_boundary'/name))
(OUT/'verification.json').write_text(json.dumps({'status':'PASS','checks':checks,'hac_tests':'PASS'},indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':checks},indent=2))
