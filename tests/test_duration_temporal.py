"""Duration clocks remain causal and identical in historical and runtime features."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import FeatureContext, FeatureManager
from historical_features import snapshot_rows
from simulator.calendar import CalendarManager
from simulator.predictor import ChampionPredictor


class DurationTemporalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'features': {'selected_features': ['wip', 'trace_elapsed', 'since_last_completion']},
                    'tasks': {'waiting_time': {'target_mode': 'raw_wait'}}}
        self.manager = FeatureManager(self.cfg)
        self.arrival = pd.Timestamp('2024-01-05 20:00', tz='UTC')

    def at(self, hours):
        return self.arrival + pd.Timedelta(hours=hours)

    def event(self, act, hour, lifecycle):
        return {'case:concept:name': 'x', 'concept:name': act,
                'time:timestamp': self.at(hour), 'lifecycle:transition': lifecycle}

    def snapshots(self, task, **kwargs):
        # B starts while A runs; C starts as B completes, before A completes.
        frame = pd.DataFrame([self.event('A', 0, 'START'), self.event('B', 2, 'START'),
                              self.event('B', 3, 'COMPLETE'), self.event('C', 3, 'START'),
                              self.event('A', 4, 'COMPLETE'), self.event('C', 5, 'COMPLETE'),
                              self.event('D', 60, 'START'), self.event('D', 61, 'COMPLETE')])
        return snapshot_rows(frame, self.cfg, task, self.manager, CalendarManager(enabled=False),
                             {}, {}, ['A', 'B', 'C', 'D'], [], {}, **kwargs)

    def test_causal_parallel_completion_and_wall_clock_runtime_parity(self):
        rows = self.snapshots('duration')
        expected = [('A', 0, 0), ('B', 7200, 7200), ('C', 10800, 0), ('D', 216000, 198000)]
        predictor = ChampionPredictor.__new__(ChampionPredictor)
        predictor.feature_mgr = self.manager
        predictor.feature_cols = ['WIP', 'Trace_Elapsed_Seconds', 'Since_Last_Completion_Seconds']
        for row, (activity, age, gap) in zip(rows, expected):
            self.assertEqual(row['Target_Activity'], activity)
            self.assertEqual(row['Trace_Elapsed_Seconds'], age)
            self.assertEqual(row['Since_Last_Completion_Seconds'], gap)
            runtime = predictor._build_feature_dict(activity, [], row['WIP'], {}, self.at(age / 3600),
                0, [], task='duration', trace_start_dt=self.arrival,
                last_completion_dt=self.at((age-gap) / 3600))
            for col in predictor.feature_cols:
                self.assertEqual(runtime[col], row[col])
        self.assertEqual(len(rows), len(expected))

    def test_waiting_dataset_unchanged(self):
        delays = pd.DataFrame([{'case_id': 'x', 'target_act': 'D',
                                'start_timestamp': self.at(60), 'predecessor_end_timestamp': self.at(5)}])
        enabled = self.snapshots('waiting_time', delay_records=delays)
        self.manager = FeatureManager({'features': {'selected_features': ['wip']}})
        disabled = self.snapshots('waiting_time', delay_records=delays)
        self.assertEqual(enabled, disabled)
        self.assertEqual(len(enabled), 1)

    def test_independent_selection_and_first_completion_fallback(self):
        for name, col in [('trace_elapsed', 'Trace_Elapsed_Seconds'),
                          ('since_last_completion', 'Since_Last_Completion_Seconds')]:
            manager = FeatureManager({'features': {'selected_features': [name]}})
            ctx = FeatureContext(task='duration', dt=self.at(2), trace_start_dt=self.arrival)
            self.assertEqual(manager.build_features(ctx), {col: 7200})
            ctx.task = 'waiting_time'
            self.assertEqual(manager.build_features(ctx), {})
        legacy = FeatureManager({'features': {'selected_features': ['wip']}})
        self.assertEqual(legacy.build_features(FeatureContext(task='duration')), {'WIP': 0})

    def test_invalid_clocks_fail_instead_of_silently_imputing(self):
        for start, last in [(None, None), (self.at(3), None), (self.arrival, self.at(3)),
                            (pd.NaT, None)]:
            with self.subTest(start=start, last=last), self.assertRaises(ValueError):
                self.manager.build_features(FeatureContext(task='duration', dt=self.at(2),
                    trace_start_dt=start, last_completion_dt=last))

    def test_model_inputs_use_duration_clocks_and_preserve_waiting_schema(self):
        predictor = ChampionPredictor.__new__(ChampionPredictor)
        predictor.activity_profiles = {}
        predictor.feature_mgr = self.manager
        predictor.feature_cols = ['WIP', 'Trace_Elapsed_Seconds', 'Since_Last_Completion_Seconds']
        predictor.max_seq_len = 1
        duration_booster, waiting_booster = Mock(), Mock()
        for booster in (duration_booster, waiting_booster):
            booster.attr.return_value = None
            booster.inplace_predict.return_value = [0.0]
        predictor.dur_models = {'models': {'A': {'type': 'xgboost', 'model': None,
            'booster': duration_booster, 'expected_cols': predictor.feature_cols}}}
        predictor.wait_models = {'models': {'A': {'type': 'xgboost', 'model': None,
            'booster': waiting_booster, 'expected_cols': ['WIP']}}}
        predictor.dur_dispatch, predictor.wait_dispatch = {}, {}
        predictor.dur_max_caps, predictor.dur_global_cap = {}, 1000
        predictor.wait_max_caps, predictor.wait_global_cap = {}, 1000
        predictor.predict_duration('A', [], 1, {}, self.at(2),
            trace_start_dt=self.arrival, last_completion_dt=self.at(1))
        self.assertEqual(duration_booster.inplace_predict.call_args.args[0].tolist(), [[1, 7200, 3600]])
        predictor.predict_waiting('A', [], 1, {}, self.at(2))
        self.assertEqual(waiting_booster.inplace_predict.call_args.args[0].tolist(), [[1]])


if __name__ == '__main__':
    unittest.main()
