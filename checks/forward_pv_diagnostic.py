"""Post-audit PV correction; never overwrite frozen policies or predictions."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compute_hedging.forward import logou_forward_K,spot_delta_to_forward_units,spot_delta_to_forward_units_pv

# Check neutrality against a central difference of actual contract PV, with fixed L.
spot=np.array([.75,1.,1.25]);dt=np.array([0.,1/12,.5]);rate=np.array([0.,.03,.08]);delta=np.array([0.,.02,.1])
k=32.66452619190628;theta=0.;sigma=.6247403836973323;bump=1e-5
delivery=logou_forward_K(spot,dt,k,theta,sigma)
value_plus=np.exp(-rate*dt)*(logou_forward_K(spot+bump,dt,k,theta,sigma)-delivery)
value_minus=np.exp(-rate*dt)*(logou_forward_K(spot-bump,dt,k,theta,sigma)-delivery)
q=spot_delta_to_forward_units_pv(delta,spot,dt,k,theta,sigma,rate)
np.testing.assert_allclose(q*(value_plus-value_minus)/(2*bump),delta,rtol=2e-4,atol=1e-10)
np.testing.assert_allclose(spot_delta_to_forward_units_pv(delta,spot,dt,k,theta,sigma,0),spot_delta_to_forward_units(delta,spot,dt,k,theta,sigma))
terminal=pd.read_csv(ROOT/'results/hedging/dynamic_forward/final_forward_terminal_errors.csv.gz')
details=pd.read_csv(ROOT/'results/hedging/dynamic_forward/final_forward_hedge_path_details.csv.gz')
factor=np.exp(.03/12);rows=[];out=ROOT/'results/reproduced/audit_corrections';out.mkdir(parents=True,exist_ok=True)
for strategy in ['oracle_rqmc','mlp_ensemble','dml_ensemble']:
    d=details[details.strategy.eq(strategy)]
    cash=d.groupby('path_uid').forward_cashflow_terminal_K.sum().reindex(terminal.path_uid).to_numpy()
    baseline=(terminal.premium_terminal_K-terminal.payoff_K).to_numpy()
    saved=terminal['hedge_error_'+strategy+'_K'].to_numpy()
    np.testing.assert_allclose(baseline+cash,saved,rtol=1e-12,atol=1e-14)
    rows.append({'strategy':strategy,'factor':factor,'historical_rmse':float(np.sqrt(np.mean(saved**2))),'pv_corrected_rmse':float(np.sqrt(np.mean((baseline+factor*cash)**2)))})
pd.DataFrame(rows).to_csv(out/'forward_pv_materiality.csv',index=False)
negative=[]
for model in ['mlp','dml']:
    preds=pd.concat([pd.read_csv(p) for p in (ROOT/'results/ml/final_evaluation/predictions').glob(model+'_n65536*__test.csv.gz')])
    for target in ['price','delta']:
        x=preds[target+'_pred'];negative.append({'model':model,'target':target,'count':len(x),'negative_count':int((x<0).sum()),'negative_pct':float(100*(x<0).mean()),'below_minus_1e_6':int((x< -1e-6).sum()),'minimum':float(x.min())})
pd.DataFrame(negative).to_csv(out/'negative_predictions.csv',index=False)
(out/'verification.json').write_text(json.dumps({'status':'PASS','contract_pv_finite_difference':'PASS','zero_rate_legacy_equivalence':'PASS','saved_cash_accounting':'PASS','posthoc':True,'forward':rows,'negative_predictions':negative},indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
