import os
import argparse
import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace, Event
from config_loader import load_config, PROJECT_ROOT
from historical_features import execution_instances
from log_utils import normalize_optional_columns


def repair_log(df, net, initial_marking, final_marking, target_acts=()):
    """Align the completion projection, reconstructing the exact paired work items.

    STARTs accompany retained synchronous completions. Other lifecycle states
    are not completions and are excluded from this simulator's work-item log.
    Model moves remain explicitly synthetic, with no invented human START.
    """
    df = normalize_optional_columns(pm4py.convert_to_dataframe(df))
    df['case:concept:name'] = df['case:concept:name'].astype(str)
    if target_acts:
        df = df[df['concept:name'].isin(target_acts)].copy()
    df['_rank'] = df['lifecycle:transition'].map({'START': 0, 'COMPLETE': 1}).fillna(2)
    df = df.sort_values(['time:timestamp', '_rank'], kind='stable').drop(columns='_rank')
    items_by_case = {}
    attrs_by_case = {}
    first_times = {}
    log_for_align = EventLog()
    for cid, grp in df.groupby('case:concept:name', sort=False):
        items = execution_instances(grp)
        if not items:
            continue
        items_by_case[cid] = items
        attrs = {'concept:name': cid}
        attrs.update({c[5:]: grp[c].iloc[0] for c in grp if c.startswith('case:') and c != 'case:concept:name'})
        attrs_by_case[cid] = attrs
        first_times[cid] = grp['time:timestamp'].iloc[0]
        log_for_align.append(Trace([Event({'concept:name': i['act'], 'time:timestamp': i['end']})
                                    for i in items], attributes={'concept:name': cid}))

    alignments = pm4py.conformance_diagnostics_alignments(log_for_align, net, initial_marking, final_marking) if log_for_align else []
    if len(alignments) != len(log_for_align):
        raise ValueError('Alignment results do not match the submitted completion traces')
    counts = {'sync': 0, 'log': 0, 'model': 0,
              'other_lifecycle': int((~df['lifecycle:transition'].isin(['START', 'COMPLETE'])).sum())}
    repaired = EventLog()
    for trace, result in zip(log_for_align, alignments):
        cid = trace.attributes['concept:name']
        if result is None or result.get('alignment') is None:
            raise ValueError(f'No valid alignment returned for case {cid}')
        items = items_by_case[cid]
        idx, curr_time = 0, first_times[cid]
        events = []
        for log_move, model_move in result['alignment']:
            item = None
            if log_move != '>>':
                if idx >= len(items) or items[idx]['act'] != log_move:
                    raise ValueError(f'Alignment/work-item mismatch for case {cid} at completion {idx}')
                item = items[idx]
                idx += 1
            if model_move is None:
                # Only a null model label denotes an invisible transition.
                if item is not None:
                    raise ValueError(f'Unexpected synchronous invisible move for case {cid}')
                continue
            if model_move == '>>':
                counts['log'] += 1
                continue
            if item is not None:
                if model_move != item['act']:
                    raise ValueError(f'Invalid synchronous labels for case {cid}')
                counts['sync'] += 1
                for row in (item['start_row'], item['row']):
                    if row is not None:
                        events.append(Event({k: v for k, v in row.items() if not k.startswith('case:') and pd.notna(v)}))
                curr_time = item['end']
            else:
                counts['model'] += 1
                events.append(Event({'concept:name': model_move, 'lifecycle:transition': 'COMPLETE',
                                     'time:timestamp': curr_time, 'org:resource': 'SYNTHETIC', 'is_synthetic': 'true'}))
        if idx != len(items):
            raise ValueError(f'Alignment did not consume all completions for case {cid}')
        # Stable ties retain the aligned COMPLETE order, including synthetic moves.
        events.sort(key=lambda e: (e['time:timestamp'], 0 if e['lifecycle:transition'] == 'START' else 1))
        repaired.append(Trace(events, attributes=attrs_by_case[cid]))
    return repaired, counts


def main():
    parser = argparse.ArgumentParser(description='Align traces to Petri net and repair event log lifecycles.')
    parser.add_argument('--config', '-c', type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_log_path = os.path.join(PROJECT_ROOT, cfg['paths']['raw_log'])
    input_pnml_path = os.path.join(PROJECT_ROOT, cfg['paths']['petri_net'])
    output_log_path = os.path.join(PROJECT_ROOT, cfg['paths']['aligned_log'])
    if not os.path.exists(input_log_path):
        raise FileNotFoundError(f'Raw log not found at: {input_log_path}')
    from temporal_split import require_artifact, stamp_artifact
    require_artifact(cfg, input_pnml_path)
    print(f'Loading log: {input_log_path}')
    df = pm4py.read_xes(input_log_path)
    net, im, fm = pm4py.read_pnml(input_pnml_path)
    repaired, counts = repair_log(df, net, im, fm, cfg.get('target_activities', []))
    print(f'Alignment summary: {counts}')
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    pm4py.write_xes(repaired, output_log_path)
    stamp_artifact(cfg, output_log_path)
    print(f'Saved aligned work-item log to {output_log_path}')


if __name__ == '__main__':
    main()
