"""Generic payload and train/runtime categorical contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features import FeatureManager, FeatureContext, validate_prediction_inputs
from feature_encoding import encode_frame, fit_encoding
from simulator.arrivals import ArrivalGenerator
from simulator.predictor import ChampionPredictor
from temporal_split import stamp_artifact


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class GenericAttributeTests(unittest.TestCase):
    def test_hospital_payload_replay_and_generated_arrivals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'log.xes'); path.touch()
            df = pd.DataFrame({'case:concept:name': ['1', '2'], 'concept:name': ['A', 'A'],
                               'time:timestamp': pd.to_datetime(['2024-01-08 09:00Z', '2024-01-08 10:00Z']),
                               'case:Department': ['001', '002'], 'case:Age': [30, 60]})
            cfg = {'paths': {'aligned_log': str(path)}, 'features': {'case_attributes': [
                {'column': 'case:Department', 'feature_name': 'Department', 'type': 'categorical'},
                {'column': 'case:Age', 'feature_name': 'Age', 'type': 'continuous'}]},
                'simulation': {'arrival': {'start_timestamp': '2024-01-08 08:00:00'}}}
            with patch('simulator.arrivals.pm4py.read_xes', return_value=df):
                for mode in ('replay', 'generative'):
                    arrivals, _ = ArrivalGenerator(cfg, num_cases=2, arrival_mode=mode).prepare_arrivals()
                    for item in arrivals:
                        self.assertIn(item['case_attrs'], [{'Department': '001', 'Age': 30.},
                                                          {'Department': '002', 'Age': 60.}])
                        self.assertEqual(item['requested_amount'], 0)
                        self.assertNotIn('RequestedAmount', item['case_attrs'])
            manager = FeatureManager({'features': {'selected_features': ['case_attributes']}})
            self.assertEqual(manager.build_features(FeatureContext(case_attrs={'Department': '001'})),
                             {'Department': '001'})

    def test_event_attribute_inputs_fail_before_io(self):
        cfg = {'features': {'selected_features': ['event_attributes'], 'event_attributes': ['Urgency']}}
        with self.assertRaisesRegex(ValueError, 'no runtime input source'):
            module('03_dataset_builder').build_dataset(cfg, 'duration')
        with self.assertRaisesRegex(ValueError, 'no runtime input source'):
            ChampionPredictor(cfg)
        with patch.object(sys, 'argv', ['train', '--task', 'duration']):
            train = module('04_train')
            with patch.object(train, 'load_config', return_value=cfg):
                with self.assertRaisesRegex(ValueError, 'no runtime input source'):
                    train.main()
        validate_prediction_inputs({'features': {'selected_features': ['wip'],
                                                  'event_attributes': ['Unused']}}, 'duration')

    def test_all_timing_engines_persist_and_reuse_categories(self):
        train = module('04_train')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'project': {'random_seed': 17},
                   'paths': {'output_base_dir': tmp, 'models_dir': tmp},
                   'features': {'selected_features': ['case_attributes', 'resource_id'],
                                'prefix_window_size': 2, 'case_attributes': [
                                    {'column': 'case:Department', 'feature_name': 'Department', 'type': 'categorical'},
                                    {'column': 'case:Age', 'feature_name': 'Age', 'type': 'continuous'}]},
                   'tasks': {'duration': {'output_file': 'data.csv', 'target_column': 'Duration_Seconds'}},
                   'model_strategy': {'dl': {'epochs': 1, 'batch_size': 8}},
                   '_split_manifest': {'split_id': 'fixture'}}
            rows = []
            for i in range(18):
                rows.append({'Case_ID': str(i), 'Instance_ID': str(i), 'Target_Activity': 'A',
                             'Prefix': '[]', 'Duration_Seconds': 10 + i, 'Sample_Weight': 1.,
                             'Partition': 'train' if i < 10 else 'validation' if i < 14 else 'test',
                             'Department': ('001' if i % 2 else '002') if i < 10 else 'NEW',
                             'Age': 20 + i, 'Resource_ID': 'UNKNOWN' if i % 2 else 'nurse'})
            dataset = Path(tmp, 'data.csv'); pd.DataFrame(rows).to_csv(dataset, index=False)
            stamp_artifact(cfg, dataset)
            for engine in ('xgboost', 'lstm', 'tcn'):
                with self.subTest(engine=engine):
                    with patch.object(train, 'load_config', return_value=cfg), \
                         patch.object(sys, 'argv', ['train', '--model_type', engine]), \
                         patch('torch.cuda.is_available', return_value=False):
                        train.main()
                    directory = Path(tmp, 'duration', 'global_' + engine)
                    metadata = json.loads((directory / 'training_metadata.json').read_text())
                    schema = metadata['feature_encoding']
                    self.assertEqual(schema['categories']['Department'], ['001', '002'])
                    self.assertNotIn('NEW', schema['categories']['Department'])
                    predictor = ChampionPredictor.__new__(ChampionPredictor)
                    predictor.cfg = cfg
                    predictor.feature_mgr = FeatureManager(cfg)
                    predictor.feature_cols = metadata['feature_columns']
                    predictor.max_seq_len = 2
                    predictor.target_acts = ['A']
                    predictor.feature_max_caps = {}
                    dispatch = json.loads((directory / 'dispatch_config.json').read_text())['dispatch']
                    suite = predictor._load_model_suite(str(directory), dispatch)
                    entry = suite['models']['A']
                    for department, worker in [('001', 'nurse'), ('NEW', 'unseen'), ('002', 'UNKNOWN')]:
                        raw = pd.DataFrame([{'Department': department, 'Age': 40., 'Resource_ID': worker}])
                        encoded = encode_frame(raw, schema)
                        if engine == 'xgboost':
                            x = pd.DataFrame({'Act_Minus_2': pd.Categorical(['NONE'], categories=['A', 'NONE']),
                                              'Act_Minus_1': pd.Categorical(['NONE'], categories=['A', 'NONE']),
                                              'Target_Activity': pd.Categorical(['A'], categories=['A'])})
                            x = pd.concat([x, encoded], axis=1)
                            direct = entry['model'].predict(x)[0]
                            expected = max(1., float(np.expm1(np.clip(direct, 0, 20))))
                        else:
                            scaler = suite['scalers']['scaler_global.pkl']
                            values = scaler.transform(encoded[metadata['feature_columns']]).astype('float32')
                            with torch.no_grad():
                                direct = entry['model'](torch.tensor([[0, 0]]), torch.tensor([0]),
                                                        torch.from_numpy(values)).item()
                            expected = max(1., float(np.expm1(np.clip(direct, 0, 15))))
                        actual = predictor._predict_time(suite, dispatch, 'A', [], 1, {},
                                                         pd.Timestamp('2024-01-08'), 0, [], 600,
                                                         worker_id=worker,
                                                         case_attrs={'Department': department, 'Age': 40.})
                        self.assertAlmostEqual(actual, expected, places=4)
