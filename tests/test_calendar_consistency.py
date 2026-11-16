"""Strict working-time targets and preservation of observed historical context."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import FeatureManager
from historical_features import snapshot_rows
from simulator.calendar import CalendarManager


def event(case, act, hour, lifecycle):
    return {'case:concept:name': case, 'concept:name': act,
            'time:timestamp': pd.Timestamp('2024-01-08') + pd.Timedelta(hours=hour),
            'lifecycle:transition': lifecycle, 'org:resource': 'w'}


class CalendarConsistencyTests(unittest.TestCase):
    def test_strict_overlap_and_expansion(self):
        for start, end, hours, elapsed in [
            ('2024-01-08 09:00', '2024-01-08 11:00', 2, 2),
            ('2024-01-08 19:00', '2024-01-08 21:00', 1, 1),
            ('2024-01-05 19:00', '2024-01-08 09:00', 2, 62),
            ('2024-01-08 21:00', '2024-01-08 22:00', 0, 0),
        ]:
            with self.subTest(start=start):
                s, e = pd.Timestamp(start), pd.Timestamp(end)
                cal = CalendarManager(ref_dt=s)
                for allow_zero in (False, True):
                    self.assertEqual(cal.calculate_net_working_seconds(s, e, allow_zero=allow_zero), hours*3600)
                self.assertEqual(cal.calculate_calendar_duration(0, hours*3600), elapsed*3600)

    def test_calendar_resolution_and_disabled_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'role_mapping.json')
            path.write_text(json.dumps({'activity_to_role': {'A': 'r'}, 'role_calendars': {'r': {'hour_end': 22}}}))
            settings = {'use_discovered_role_calendars': True}
            cfg = {'simulation': {'calendar': settings}}
            s, e = pd.Timestamp('2024-01-08 19:00'), pd.Timestamp('2024-01-08 21:00')
            for overrides, expected in [({}, 2), ({'r': {'hour_end': 20}}, 1),
                                        ({'r': {'hour_end': 20}, 'A': {'hour_end': 23}}, 2)]:
                settings['per_role'] = overrides
                cal = CalendarManager.from_config(cfg, ref_dt=s, role_mapping_path=path)
                self.assertEqual(cal.calculate_net_working_seconds(s, e, 'A'), expected*3600)
                self.assertEqual(cal.calculate_calendar_duration(0, expected*3600, 'A'), expected*3600)
                self.assertEqual(cal.calculate_net_working_seconds(s, e, 'unmapped'), 3600)
            for settings in ({'enabled': False}, {'per_role': {'A': {'enabled': False}}}):
                cal = CalendarManager.from_config({'simulation': {'calendar': settings}}, ref_dt=s, role_mapping_path=path)
                self.assertEqual(cal.calculate_net_working_seconds(s, e, 'A'), 7200)
                self.assertEqual(cal.calculate_calendar_duration(0, 7200, 'A'), 7200)

    def fixture(self):
        # B starts while excluded A is active; C follows A in the same case.
        rows = [event('x', 'A', 21, 'START'), event('y', 'B', 21.5, 'START'),
                event('x', 'A', 22, 'COMPLETE'), event('y', 'B', 22.5, 'COMPLETE'),
                event('x', 'C', 33, 'START'), event('x', 'C', 34, 'COMPLETE')]
        cfg = {'features': {'selected_features': ['wip', 'ac_wip', 'role_occupancy', 'prev_duration', 'trace_history']},
               'tasks': {'duration': {'filter_zero_duration': False}, 'waiting_time': {}}}
        roles = {'__role_names__': ['r'], 'role_to_workers': {'r': ['w']}, 'worker_concurrencies': {'w': 2}}
        return dict(df=pd.DataFrame(rows), cfg=cfg, manager=FeatureManager(cfg),
                    calendar=CalendarManager(per_role={'B': {'enabled': False}}), role_data=roles,
                    activity_types={}, activities=['A', 'B', 'C'], tracked=['A', 'B', 'C'], imputation={})

    def test_zero_overlap_preserves_context_and_waiting(self):
        kwargs = self.fixture()
        summary = {}
        rows = snapshot_rows(task='duration', diagnostics=summary, **kwargs)
        self.assertEqual(summary['excluded'], {'zero_working_time_overlap': 1})
        self.assertEqual([r['Target_Activity'] for r in rows], ['B', 'C'])
        b, c = rows
        self.assertEqual((b['WIP'], b['Role_0_OC']), (2, 1))
        self.assertEqual(c['Prefix'], '["A"]')
        self.assertEqual(c['Prev_Proc_Time'], 0)
        self.assertEqual(c['Role_0_OC'], .5)  # both earlier executions released
        delays = pd.DataFrame([{'case_id': 'x', 'target_act': 'A',
                                'start_timestamp': pd.Timestamp('2024-01-08 21:00'),
                                'predecessor_end_timestamp': pd.Timestamp('2024-01-08 21:00')}])
        waits = snapshot_rows(task='waiting_time', delay_records=delays, **kwargs)
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]['Target_Activity'], 'A')
        self.assertEqual(waits[0]['Wait_Time_Seconds'], 0)

    def test_instantaneous_missing_start_and_synthetic_unchanged(self):
        kwargs = self.fixture()
        rows = [event('x', 'A', 21, 'START'), event('x', 'A', 21, 'COMPLETE'),
                event('y', 'A', 22, 'COMPLETE'),
                dict(event('z', 'A', 23, 'COMPLETE'), is_synthetic=True)]
        kwargs['df'] = pd.DataFrame(rows)
        summary = {}
        result = snapshot_rows(task='duration', diagnostics=summary, **kwargs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['Duration_Seconds'], 0)
        self.assertEqual(summary['excluded'], {'missing_start': 1, 'synthetic': 1})

    def test_builder_summary_and_empty_dataset_failure(self):
        spec = importlib.util.spec_from_file_location('calendar_dataset_builder', Path(__file__).resolve().parents[1] / 'src/03_dataset_builder.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp, 'log.xes'); log.touch()
            Path(tmp, 'role_mapping.json').write_text(json.dumps({'activity_to_role': {}, 'roles': {}}))
            cfg = {'paths': {'aligned_log': str(log), 'models_dir': tmp, 'output_base_dir': tmp},
                   'features': {}, 'tasks': {'duration': {'target_column': 'Duration_Seconds',
                   'output_file': 'duration.csv', 'filter_zero_duration': False}}}
            outside = [event('x', 'A', 21, 'START'), event('x', 'A', 22, 'COMPLETE')]
            inside = [event('y', 'A', 9, 'START'), event('y', 'A', 10, 'COMPLETE')]
            output = Path(tmp, 'duration.csv')
            with patch.object(builder, 'load_or_discover_activity_profiles', return_value={'activity_types': {}}):
                with patch.object(builder.pm4py, 'read_xes', return_value=pd.DataFrame(outside)):
                    with self.assertRaisesRegex(ValueError, 'No usable duration.*zero_working_time_overlap'):
                        builder.build_dataset(cfg)
                self.assertFalse(output.exists())
                with patch.object(builder.pm4py, 'read_xes', return_value=pd.DataFrame(outside + inside)):
                    builder.build_dataset(cfg)
                summary = json.loads(Path(str(output) + '.summary.json').read_text())
                self.assertEqual(summary['excluded']['zero_working_time_overlap'], 1)
                self.assertEqual(summary['rows_after_target_filter'], 1)


if __name__ == '__main__':
    unittest.main()
