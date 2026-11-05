#!/usr/bin/env python3
"""
src/simulator/token.py

Petri Net Case Token & Transition Execution Engine.
Encapsulates individual process instance lifecycle, including:
  1. Formal PM4Py Petri net marking semantics: M' = M - Preset(t) + Postset(t)
  2. XOR routing decision-point queries via ChampionPredictor
  3. Shared worker acquisition and release
  4. Chapela-Campa Eclipse-Aware residual waiting time: max(0, W_ML - ext_delay) * loop_decay
  5. Calendar-bounded working duration shift calculation
  6. Contextual milestone and attribute progression tracking
"""

import random
from collections import defaultdict
from pm4py.objects.petri_net import semantics


class Token:
    """
    Case Instance (Parity with Legacy token_LSTM.py / event_trace.py).
    Represents an active case token flowing through the Petri net process model.
    """
    def __init__(self, case_id, arrival_offset, requested_amount=0.0, process=None, init_ms=None, ms_states=None, inter_arrival_seconds=0.0, case_attrs=None):
        self.case_id = str(case_id)
        self.arrival_offset = arrival_offset
        self.process = process
        self.env = process.env
        self.case_attrs = dict(case_attrs or {})
        if requested_amount and "RequestedAmount" not in self.case_attrs:
            self.case_attrs["RequestedAmount"] = float(requested_amount)
        self.requested_amount = float(self.case_attrs.get("RequestedAmount", requested_amount))
        self.inter_arrival_seconds = float(inter_arrival_seconds)
        self.arrival_dt = self.process.calendar.to_datetime(self.arrival_offset)

        # Prefix and milestone state tracking
        self.prefix = []
        ms_prefixes = self.process.ms_prefixes
        cfg_init_ms = set(self.process.cfg.get("preprocessing", {}).get("initial_milestones", []))
        if init_ms is not None:
            self.milestones_seen = set(init_ms)
        elif cfg_init_ms:
            self.milestones_seen = set(cfg_init_ms)
        elif any(m.startswith("A_") for m in ms_prefixes):
            self.milestones_seen = {"A_SUBMITTED", "A_PARTLYSUBMITTED"}
        else:
            self.milestones_seen = set()

        self.offer_count = float(
            sum(1 for m in self.milestones_seen if m.startswith("O_"))
            if (any(m.startswith("O_") for m in self.milestones_seen) or any(p.startswith("O_") for p in ms_prefixes))
            else 0.0
        )
        self.has_offer = 1.0 if self.offer_count > 0 else 0.0

        self.events_done = 0
        self.prev_act = None
        self.prev_dur = 0.0
        self.dp_counts = defaultdict(int)

        # Initialize marking at source place(s)
        self.marking = self.process.im.copy()

    def update_milestones(self, act):
        """Updates milestone progress context dynamically based on config rules."""
        ms_prefixes = self.process.ms_prefixes
        if not ms_prefixes:
            return

        prog = self.process.cfg.get("preprocessing", {}).get("milestone_progression", {})
        if prog and act in prog:
            item = prog[act]
            if isinstance(item, dict):
                for m, prob in item.items():
                    if random.random() < float(prob):
                        self.milestones_seen.add(m)
                        if any(p.startswith("O_") for p in ms_prefixes) and m.startswith("O_"):
                            self.offer_count += 1.0
                            self.has_offer = 1.0
            elif isinstance(item, (list, tuple, set)):
                for m in item:
                    self.milestones_seen.add(m)
                    if any(p.startswith("O_") for p in ms_prefixes) and m.startswith("O_"):
                        self.offer_count += 1.0
                        self.has_offer = 1.0

    def _consume_tokens(self, transition):
        """Removes input tokens for transition from marking."""
        for arc in transition.in_arcs:
            if arc.source in self.marking:
                self.marking[arc.source] -= arc.weight
                if self.marking[arc.source] <= 0:
                    del self.marking[arc.source]

    def _produce_tokens(self, transition):
        """Deposits output tokens for transition into marking."""
        for arc in transition.out_arcs:
            self.marking[arc.target] = self.marking.get(arc.target, 0) + arc.weight

    def _resolve_conflict_group(self, cand_transitions):
        """Resolves XOR choice among conflicting transitions sharing an input place."""
        if len(cand_transitions) == 1:
            return cand_transitions[0]

        decision_places = sorted(
            [p for p in self.marking if len(p.out_arcs) >= 2 and any(arc.source == p for t in cand_transitions for arc in t.in_arcs)],
            key=lambda p: p.name
        )
        if decision_places:
            dp = decision_places[0].name
            loop_count = float(self.dp_counts[dp])
            self.dp_counts[dp] += 1
            predicted_target = self.process.predictor.predict_routing(
                dp, self.prefix, self.requested_amount, self.milestones_seen,
                self.offer_count, self.has_offer,
                curr_dt=self.arrival_dt,
                loop_count=loop_count,
                inter_arrival_seconds=self.inter_arrival_seconds,
                case_attrs=self.case_attrs
            )
            matched_t = None
            if predicted_target is not None:
                for cand_t in cand_transitions:
                    tgt = self.process.branch_target_map.get((dp, cand_t.name))
                    if tgt == predicted_target or cand_t.name == predicted_target or (cand_t.label is not None and cand_t.label == predicted_target):
                        matched_t = cand_t
                        break
            if matched_t:
                return matched_t

            # Fallback using empirical branching probabilities
            dp_manifest = self.process.predictor.routing_manifest.get(dp, {})
            emp_probs = dp_manifest.get("probabilities", {})
            weights = []
            for cand_t in cand_transitions:
                cand_tgt = self.process.branch_target_map.get((dp, cand_t.name), cand_t.name)
                w = emp_probs.get(cand_tgt, emp_probs.get(cand_t.name, 1.0))
                weights.append(max(0.01, float(w)))
            return random.choices(cand_transitions, weights=weights, k=1)[0]
        else:
            return random.choice(cand_transitions)

    def _partition_and_resolve_transitions(self, enabled):
        """Partitions enabled transitions into independent conflict sets and selects one transition per conflict set.
        Transitions sharing an input place are in XOR conflict; transitions with disjoint input places run concurrently (AND-split)."""
        # Build conflict graph
        selected = []
        remaining = list(enabled)
        while remaining:
            first = remaining.pop(0)
            group = [first]
            first_in = set(arc.source for arc in first.in_arcs)
            idx = 0
            while idx < len(remaining):
                cand = remaining[idx]
                cand_in = set(arc.source for arc in cand.in_arcs)
                if first_in & cand_in:
                    group.append(cand)
                    first_in.update(cand_in)
                    remaining.pop(idx)
                else:
                    idx += 1
            chosen = self._resolve_conflict_group(group)
            selected.append(chosen)
        return selected

    def _execute_work_item(self, chosen_t, on_complete):
        """SimPy process executing an observable work item activity concurrently."""
        act = chosen_t.label
        self.events_done += 1

        act_occurrences = self.prefix.count(act)
        loop_decay = 0.5 ** act_occurrences if act_occurrences > 0 else 1.0

        # 3. Chapela-Campa Transition-Arc Delay (specific) or Ex-Ante Extraneous Delay (generic fallback)
        ext_delay = 0.0
        if self.prev_act and (self.prev_act, act) in self.process.external_delays:
            ext_delay = self.process.sample_external_delay(self.prev_act, act) * loop_decay
            if ext_delay > 0:
                yield self.env.timeout(ext_delay)
        elif self.process.has_ex_ante_delay(act):
            ext_delay = self.process.sample_ex_ante_delay(act) * loop_decay
            if ext_delay > 0:
                yield self.env.timeout(ext_delay)

        # 4. Worklist Waiting Time (ML expectation: batching & uncoupled backlog)
        role_oc = self.process.workers.get_role_occupancy(self.process.target_activities)
        curr_dt = self.process.calendar.to_datetime(self.env.now)
        if self.process.mode == "hybrid_residual":
            w_ml = self.process.predictor.predict_waiting(
                act, self.prefix, self.process.active_cases_count,
                self.process.ac_wip, curr_dt, self.requested_amount,
                self.milestones_seen, self.offer_count, self.has_offer, role_oc,
                prev_dur=self.prev_dur,
                case_attrs=self.case_attrs
            )
            w_worklist = max(0.0, w_ml * loop_decay - ext_delay)
            if w_worklist > 0:
                yield self.env.timeout(w_worklist)
        else:
            w_worklist = 0.0

        act_type = self.process.activity_profiles.get(act, "HUMAN_WORK_ITEM")
        if act_type == "AUTOMATED_TASK":
            # Automated task: runs 24/7 without human worker allocation; duration is 0.0s
            total_wait = ext_delay + w_worklist
            t_comp = self.env.now
            self._produce_tokens(chosen_t)
            comp_ts = self.process.calendar.to_datetime(t_comp).strftime("%Y-%m-%d %H:%M:%S")
            self.process.record_event({
                "case_id": self.case_id,
                "activity": act,
                "start_timestamp": comp_ts,
                "complete_timestamp": comp_ts,
                "resource_id": "SYSTEM",
                "duration_seconds": 0.0,
                "wait_time_seconds": round(total_wait, 2),
                "queue_wait_seconds": 0.0,
                "wip": self.process.active_cases_count
            })
            self.prefix.append(act)
            self.prev_act = act
            self.prev_dur = 0.0
            on_complete()
            return

        # 5. Calendar Working Hours Gate: If office is closed, ticket waits in worklist queue until shift begins
        cal = self.process.calendar.get_calendar(act)
        if cal.enabled:
            curr_dt_shift = cal.to_datetime(self.env.now)
            shift_gap = cal.get_gap_to_next_work_window(curr_dt_shift)
            if shift_gap > 0:
                yield self.env.timeout(shift_gap)
        else:
            shift_gap = 0.0

        # 6. Physical Worker Acquisition at Start of Execution
        t_req = self.env.now
        worker_event = self.process.workers.request_worker(act)
        worker_id = yield worker_event
        t_start = self.env.now
        w_queue = max(0.0, t_start - t_req)
        total_wait = ext_delay + w_worklist + shift_gap + w_queue

        # 7. Execution Duration with Calendar Working Hours (Per-Role or Global)
        curr_dt = self.process.calendar.to_datetime(t_start)
        role_oc = self.process.workers.get_role_occupancy(self.process.target_activities)

        self.process.ac_wip[act] += 1
        dur_pred = self.process.predictor.predict_duration(
            act, self.prefix, self.process.active_cases_count,
            self.process.ac_wip, curr_dt, self.requested_amount,
            self.milestones_seen, self.offer_count, self.has_offer, role_oc,
            worker_id=worker_id,
            prev_dur=self.prev_dur,
            case_attrs=self.case_attrs
        )

        actual_dur = self.process.calendar.calculate_calendar_duration(
            self.env.now, dur_pred, role_or_act=act
        )
        yield self.env.timeout(actual_dur)
        t_comp = self.env.now

        # 8. Release Worker
        self.process.ac_wip[act] = max(0, self.process.ac_wip[act] - 1)
        self.process.workers.release_worker(worker_id, act)

        # 9. Deposit Output Tokens into Petri Net
        self._produce_tokens(chosen_t)

        # 10. Log Event
        start_ts = self.process.calendar.to_datetime(t_start).strftime("%Y-%m-%d %H:%M:%S")
        comp_ts = self.process.calendar.to_datetime(t_comp).strftime("%Y-%m-%d %H:%M:%S")
        self.process.record_event({
            "case_id": self.case_id,
            "activity": act,
            "start_timestamp": start_ts,
            "complete_timestamp": comp_ts,
            "resource_id": worker_id,
            "duration_seconds": round(actual_dur, 2),
            "wait_time_seconds": round(total_wait, 2),
            "queue_wait_seconds": round(w_queue, 2),
            "wip": self.process.active_cases_count
        })

        self.prefix.append(act)
        self.prev_act = act
        self.prev_dur = dur_pred

        # Advance milestone state dynamically based on actual executed activity
        self.update_milestones(act)
        self.has_offer = 1.0 if self.offer_count > 0 else 0.0

        on_complete()

    def run(self):
        """
        SimPy Generator: Simulates case token lifecycle across Petri net places,
        transitions, and worker contention queues using token marking semantics.
        Supports true concurrent execution of parallel branches (AND-splits) and
        warns if an unsound Petri net encounters a deadlock before final marking.
        """
        yield self.env.timeout(self.arrival_offset)

        self.process.active_cases_count += 1
        active_tasks = 0
        marking_event = self.env.event()

        def _on_task_complete():
            nonlocal active_tasks, marking_event
            active_tasks = max(0, active_tasks - 1)
            if not marking_event.triggered:
                marking_event.succeed()

        # Step through Petri Net until final marking or safety ceiling
        silent_chain_guard = 0
        while (self.marking or active_tasks > 0) and self.events_done < self.process.max_events:
            # Check if clean termination is reached
            if self.marking == self.process.fm and active_tasks == 0:
                break

            # 1. Identify all enabled transitions (both silent and observable)
            enabled = sorted(
                list(semantics.enabled_transitions(self.process.net, self.marking)),
                key=lambda t: t.name
            )

            if not enabled:
                if active_tasks > 0:
                    # Parallel branches are in flight; wait for at least one branch to complete
                    marking_event = self.env.event()
                    yield marking_event
                    continue
                else:
                    # Deadlock: No running tasks and no enabled transitions before final marking
                    if self.marking and self.marking != self.process.fm:
                        deadlock_places = sorted([p.name for p in self.marking])
                        print(f"  [!] Warning: Case {self.case_id} encountered a Petri net deadlock in place(s) {deadlock_places} before reaching final marking. Trace terminated safely.")
                    break

            # 2. Partition enabled transitions into independent sets (AND-splits run in parallel, XOR splits resolved)
            selected_transitions = self._partition_and_resolve_transitions(enabled)

            # 3. Execute selected transitions (silent transitions & milestones fire in 0s; work items spawn SimPy tasks)
            immediate_progress = False
            for chosen_t in selected_transitions:
                if chosen_t.label is None:
                    self.marking = semantics.execute(chosen_t, self.process.net, self.marking)
                    silent_chain_guard += 1
                    if silent_chain_guard < 500:
                        immediate_progress = True
                    continue

                silent_chain_guard = 0
                act = chosen_t.label

                # Check activity profile: INSTANT_GATEWAY vs. AUTOMATED_TASK vs. HUMAN_WORK_ITEM
                act_type = getattr(self.process, "activity_profiles", {}).get(act)
                if not act_type:
                    proc_prefixes = self.process.proc_prefixes
                    ms_prefixes = self.process.ms_prefixes
                    if self.process.target_activities:
                        is_work_item = (act in self.process.target_activities)
                    elif proc_prefixes:
                        is_work_item = act.startswith(proc_prefixes)
                    elif ms_prefixes:
                        is_work_item = not act.startswith(ms_prefixes)
                    else:
                        is_work_item = True
                    act_type = "HUMAN_WORK_ITEM" if is_work_item else "INSTANT_GATEWAY"

                if act_type == "INSTANT_GATEWAY":
                    # Instantaneous lifecycle milestone (e.g. A_ / O_ state transition or instant gateway)
                    self.events_done += 1
                    self.milestones_seen.add(act)
                    ms_prefixes = self.process.ms_prefixes
                    if ms_prefixes and any(p.startswith("O_") for p in ms_prefixes) and act.startswith("O_"):
                        self.offer_count += 1.0
                        self.has_offer = 1.0
                    t_event = self.env.now
                    self.marking = semantics.execute(chosen_t, self.process.net, self.marking)
                    event_ts = self.process.calendar.to_datetime(t_event).strftime("%Y-%m-%d %H:%M:%S")
                    self.process.record_event({
                        "case_id": self.case_id,
                        "activity": act,
                        "start_timestamp": event_ts,
                        "complete_timestamp": event_ts,
                        "resource_id": "SYSTEM",
                        "duration_seconds": 0.0,
                        "wait_time_seconds": 0.0,
                        "queue_wait_seconds": 0.0,
                        "wip": self.process.active_cases_count
                    })
                    self.prefix.append(act)
                    self.prev_act = act
                    self.prev_dur = 0.0
                    immediate_progress = True
                else:
                    # Observable work item or automated delayed task: consume input tokens and launch SimPy process
                    self._consume_tokens(chosen_t)
                    active_tasks += 1
                    self.env.process(self._execute_work_item(chosen_t, _on_task_complete))

            # If silent transitions or milestones advanced the marking, immediately re-check enabled transitions at t=env.now
            if immediate_progress:
                continue

            if active_tasks > 0 and not semantics.enabled_transitions(self.process.net, self.marking):
                marking_event = self.env.event()
                yield marking_event

        self.process.active_cases_count = max(0, self.process.active_cases_count - 1)

