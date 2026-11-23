import sys, json, importlib.util, copy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
import numpy as np
import pandas as pd
import pm4py
import joblib
from sklearn.metrics import accuracy_score, f1_score
from config_loader import load_config
from temporal_split import split_manifest
from simulator.process import SimulationProcess
root = Path(__file__).resolve().parents[3]
output = root / 'data/experiments/routing_temporal'
output.mkdir(parents=True, exist_ok=True)
cfg = load_config(str(root/'config.yaml'))
split = split_manifest(cfg)
spec = importlib.util.spec_from_file_location('routing', root/'src/05_train_routing.py')
routing = importlib.util.module_from_spec(spec); spec.loader.exec_module(routing)
print('Loading held-out validation cases', flush=True)
frame = pm4py.read_xes(cfg['paths']['routing_train_log'])
ids = {cid for cid, info in split['cases'].items() if info['partition']=='validation'}
frame = frame[frame['case:concept:name'].astype(str).isin(ids)].copy()
net, im, fm = pm4py.read_pnml(cfg['paths']['routing_petri_net'])
# Reuse discovery as called by Step 05.
xor = routing.discover_xor_decision_points(cfg['paths']['routing_petri_net'])
with patch.object(routing.pm4py, 'read_xes', return_value=frame):
    samples = routing.extract_decision_samples_via_alignment('validation', cfg['paths']['routing_petri_net'], xor,
        case_attrs_cfg=cfg['features'].get('case_attributes', []), routing_mode=cfg['tasks']['routing']['mode'],
        include_counts=cfg['features'].get('trace_history', {}).get('include_counts', False))
report = {'evaluation': 'shared temporal validation cases; deterministic argmax classification; mean predicted branch probabilities',
          'validation_cases': len(ids), 'results': {}, 'simulation': {}}
for label, directory in [('baseline', output/'baseline_models'), ('temporal', root/'models/routing')]:
    manifest = json.loads((directory/'routing_decisions.json').read_text())
    results = {}
    for place, df in samples.items():
        info = manifest['decision_points'].get(place)
        if info is None or df.empty: continue
        df = df.copy(); df['hour'] = df.start_hour; df['weekday'] = df.start_weekday
        observed = df.chosen_target.value_counts(normalize=True).to_dict()
        if info['strategy']=='machine_learning':
            model = joblib.load(directory/info['model_file'])
            x = df[info['features']].copy()
            for col, cats in info.get('cat_mappings', {}).items():
                x[col] = pd.Categorical(x[col], categories=cats).codes
            reverse = {v:k for k,v in info['label_encoding'].items()}
            pred = [reverse[i] for i in model.predict(x)]
            probabilities = {reverse[int(i)]: float(v) for i,v in zip(model.classes_, model.predict_proba(x).mean(axis=0))}
        else:
            probabilities = info['probabilities']
            pred = [max(probabilities, key=probabilities.get)] * len(df)
        results[place] = {'samples': len(df), 'strategy':info['strategy'],
            'accuracy': float(accuracy_score(df.chosen_target, pred)),
            'macro_f1': float(f1_score(df.chosen_target, pred, average='macro', zero_division=0)),
            'observed_probabilities': observed, 'mean_predicted_probabilities': probabilities}
    report['results'][label] = results
(output/'comparison.json').write_text(json.dumps(report, indent=2))
print('Classification comparison saved', flush=True)
for label, directory in [('baseline', output/'baseline_models'), ('temporal', root/'models/routing')]:
    run_cfg = copy.deepcopy(cfg)
    dest = output/label; dest.mkdir(exist_ok=True)
    run_cfg['paths']['routing_models_dir'] = str(directory)
    run_cfg['paths']['output_base_dir'] = str(dest)
    run_cfg['simulation']['output_simulated_log'] = str(dest/'simulated.csv')
    run_cfg['simulation']['output_simulated_xes'] = str(dest/'simulated.xes')
    print(f'Simulating {label}: 100 identical earliest replay arrivals, seed 42', flush=True)
    try:
        sim = SimulationProcess(run_cfg, num_cases=100, arrival_mode='replay', export_xes=False)
        metrics = sim.run()
        events = pd.read_csv(sim.sim_log_path)
        completed = events[events.complete_timestamp.notna()].copy()
        paths = completed.groupby('case_id', sort=False)['activity'].agg(tuple).value_counts()
        report['simulation'][label] = {'status': 'ok', 'cases':100, 'seed':42,
            'activity_counts': completed.activity.value_counts().to_dict(),
            'path_counts': [{'path':list(k), 'count':int(v)} for k,v in paths.items()],
            'metrics':metrics}
    except Exception as exc:
        report['simulation'][label] = {'status':'failed', 'error':repr(exc)}
        print(f'Simulation failed: {exc}', flush=True)
    (output/'comparison.json').write_text(json.dumps(report, indent=2, default=str))
