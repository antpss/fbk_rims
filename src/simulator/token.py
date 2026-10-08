#!/usr/bin/env python3
"""
src/simulator/token.py

Petri Net Case Token & Transition Execution Engine.
Encapsulates individual process instance lifecycle, including:
  1. Formal PM4Py Petri net marking semantics: M' = M - Preset(t) + Postset(t)
  2. XOR routing decision-point queries via ChampionPredictor
  3. Shared worker acquisition and release
  4. Hybrid residual waiting time calculation: max(0, W_ML - W_queue)
  5. Calendar-bounded working duration shift calculation
  6. Contextual milestone and attribute progression tracking
"""

import random
from pm4py.objects.petri_net import semantics


class Token:
    """
    Case Instance (Parity with Legacy token_LSTM.py / event_trace.py).
    Represents an active case token flowing through the Petri net process model.
    """
    def __init__(self, case_id, arrival_offset, requested_amount, process, init_ms=None, ms_states=None):
        self.case_id = str(case_id)
        self.arrival_offset = arrival_offset
        self.requested_amount = float(requested_amount)
        self.process = process
        self.env = process.env

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

        self.offer_count = (
            sum(1 for m in self.milestones_seen if m.startswith("O_"))
            if ("Offer_Count" in self.process.predictor.feature_cols or "O_" in ms_prefixes)
            else 0
        )
        self.has_offer = 1 if self.offer_count > 0 else 0

        self.ms_states = ms_states or []
        self.events_done = 0
        self.prev_act = None

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
            elif isinstance(item, (list, tuple, set)):
                self.milestones_seen.update(item)
        elif not prog and any(m.startswith("A_") for m in ms_prefixes):
            if act == "W_Complete application":
                self.milestones_seen.update(["A_ACCEPTED", "O_SELECTED", "A_FINALIZED", "O_CREATED", "O_SENT"])
            elif act == "W_Call after offers":
                self.milestones_seen.add("O_SENT_BACK")
            elif act == "W_Validate application":
                self.milestones_seen.update(["A_APPROVED", "A_REGISTERED", "A_ACTIVATED", "O_ACCEPTED"])
            elif act == "W_Handle leads":
                self.milestones_seen.add("A_PREACCEPTED")

    def run(self):
        """
        SimPy Generator: Simulates case token lifecycle across Petri net places,
        transitions, and worker contention queues using PM4Py token marking semantics.
        """
        yield self.env.timeout(self.arrival_offset)

        self.process.active_cases_count += 1

        # Step through Petri Net until final marking or safety ceiling
        while self.marking and self.marking != self.process.fm and self.events_done < self.process.max_events:
            enabled = list(semantics.enabled_transitions(self.process.net, self.marking))
            if not enabled:
                # Deadlock or process termination reached
                break

            # 1. Resolve Transition Choice at Decision Points
            if len(enabled) == 1:
                chosen_t = enabled[0]
            else:
                # XOR Decision Point: identify place in marking with out-degree >= 2
                decision_places = [p for p in self.marking if len(p.out_arcs) >= 2]
                if decision_places:
                    dp = decision_places[0].name
                    predicted_target = self.process.predictor.predict_routing(
                        dp, self.prefix, self.requested_amount, self.milestones_seen,
                        self.offer_count, self.has_offer,
                        curr_dt=self.process.calendar.to_datetime(self.env.now)
                    )
                    # Match predicted target to an enabled transition
                    matched_t = None
                    for cand_t in enabled:
                        tgt = self.process.branch_target_map.get((dp, cand_t.name))
                        if tgt == predicted_target:
                            matched_t = cand_t
                            break
                    chosen_t = matched_t if matched_t else enabled[0]
                else:
                    chosen_t = enabled[0]

            # 2. Check if chosen transition is Silent vs. Observable Work Item
            if chosen_t.label is None:
                # Silent transition: instant state transition
                self.marking = semantics.execute(chosen_t, self.process.net, self.marking)
                continue

            # Observable Work Item Activity
            act = chosen_t.label
            self.events_done += 1

            # 3. External Uncoupled Delay (e.g. customer postal / response transit)
            if self.prev_act and (self.prev_act, act) in self.process.external_delays:
                ext_delay = self.process.sample_external_delay(self.prev_act, act)
                if ext_delay > 0:
                    yield self.env.timeout(ext_delay)

            # 4. Physical Queue Contention (SimPy)
            t_req = self.env.now
            worker_event = self.process.workers.request_worker(act)
            worker_id = yield worker_event
            t_acquired = self.env.now
            w_queue = max(0.0, t_acquired - t_req)

            # 5. Predicted Waiting Time & Agnostic Hybrid Residual
            role_oc = self.process.workers.get_role_occupancy(self.process.target_activities)
            curr_dt = self.process.calendar.to_datetime(self.env.now)
            if self.process.mode != "pure_physics":
                w_ml = self.process.predictor.predict_waiting(
                    act, self.prefix, self.process.active_cases_count,
                    self.process.ac_wip, curr_dt, self.requested_amount,
                    self.milestones_seen, self.offer_count, self.has_offer, role_oc
                )
            else:
                w_ml = 0.0

            if self.process.mode == "hybrid_residual":
                w_residual = max(0.0, w_ml - w_queue)
                if w_residual > 0:
                    yield self.env.timeout(w_residual)
                total_wait = w_queue + w_residual
            elif self.process.mode == "pure_ml":
                yield self.env.timeout(w_ml)
                total_wait = w_ml
            elif self.process.mode == "pure_physics":
                total_wait = w_queue

            # 6. Execution Duration with Calendar Working Hours (Per-Role or Global)
            t_start = self.env.now
            self.process.ac_wip[act] += 1
            dur_pred = self.process.predictor.predict_duration(
                act, self.prefix, self.process.active_cases_count,
                self.process.ac_wip, curr_dt, self.requested_amount,
                self.milestones_seen, self.offer_count, self.has_offer, role_oc
            )

            actual_dur = self.process.calendar.calculate_calendar_duration(
                self.env.now, dur_pred, role_or_act=act
            )
            yield self.env.timeout(actual_dur)
            t_comp = self.env.now

            # 7. Release Worker
            self.process.ac_wip[act] = max(0, self.process.ac_wip[act] - 1)
            self.process.workers.release_worker(worker_id)

            # 8. Fire Transition in Petri Net Marking
            self.marking = semantics.execute(chosen_t, self.process.net, self.marking)

            # 9. Log Event
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

            # Advance milestone state if available from case history, else apply progression
            if self.events_done < len(self.ms_states):
                self.milestones_seen = set(self.ms_states[self.events_done])
            else:
                self.update_milestones(act)

            ms_prefixes = self.process.ms_prefixes
            self.offer_count = (
                sum(1 for m in self.milestones_seen if m.startswith("O_"))
                if ("Offer_Count" in self.process.predictor.feature_cols or "O_" in ms_prefixes)
                else 0
            )
            self.has_offer = 1 if self.offer_count > 0 else 0

        self.process.active_cases_count = max(0, self.process.active_cases_count - 1)

