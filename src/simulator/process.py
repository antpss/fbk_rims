#!/usr/bin/env python3
"""
src/simulator/process.py

Organizational Simulation Engine & Multi-Skilled Human Worker Pool.
Implements:
  1. SharedWorkerPool: Eliminates phantom staff by modeling shared multi-skilling
     and refractory inter-ticket human latency cooldowns.
  2. SimulationProcess: Coordinates the SimPy environment, Petri net execution,
     arrival queues, and output serialization (Legacy parity with checking_process.py).
"""

import os
import json
import time
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import scipy.stats as st
import simpy
import pm4py

try:
    from config_loader import PROJECT_ROOT
except ImportError:
    from src.config_loader import PROJECT_ROOT
from .calendar import CalendarManager
from .predictor import ChampionPredictor
from .arrivals import ArrivalGenerator
from .token import Token
from .metrics import SimulationEvaluator


class SharedWorkerPool:
    """
    Manages the distinct enterprise human resources across all skills.
    Eliminates phantom staff by ensuring each individual human worker can only
    perform one task at a time across all business activities.
    Supports Kolmogorov-Smirnov fitted refractory latency cooldowns.
    """
    def __init__(self, env, role_mapping_path, latency_cfg=None, calendar=None):
        self.env = env
        self.latency_cfg = latency_cfg or {}
        self.calendar = calendar
        with open(role_mapping_path, "r") as f:
            data = json.load(f)

        self.activity_to_role = data.get("activity_to_role", {})
        self.role_to_workers = data.get("role_to_workers", {})
        self.role_names = data.get("__role_names__") or list(data.get("roles", {}).keys()) or sorted(list(set(self.activity_to_role.values())))
        self.role_capacities = data.get("role_capacities", {r: 1 for r in self.role_names})

        self.all_workers = data.get("__all_resources__", [])
        if not self.all_workers:
            self.all_workers = sorted(list(set.union(*[set(data[k]) for k in data if not k.startswith("__")])))

        self.activity_pools = {k: data[k] for k in data if not k.startswith("__")}
        self.worker_concurrencies = data.get("worker_concurrencies", {})
        self.resource_capacities = data.get("__resource_capacities__", {})
        self.worker_active_count = {w: 0 for w in self.all_workers}
        self.worker_busy = {w: False for w in self.all_workers}
        self.worker_activities = {w: [] for w in self.all_workers}
        self.worker_activity = {w: None for w in self.all_workers}
        self.worker_idle_since = {w: 0.0 for w in self.all_workers}
        self.pending_requests = []
        self._shift_alarm_active = False

    def _get_capacity(self, worker_id):
        if worker_id in self.worker_concurrencies:
            return int(self.worker_concurrencies[worker_id])
        return int(self.resource_capacities.get(worker_id, 1))

    def _is_worker_free(self, worker_id):
        return self.worker_active_count[worker_id] < self._get_capacity(worker_id)

    def _is_working_now(self):
        if not self.calendar or not getattr(self.calendar, "enabled", False):
            return True
        curr_dt = self.calendar.to_datetime(self.env.now)
        return self.calendar.is_working_hour(curr_dt)

    def _ensure_shift_alarm(self):
        if not self.calendar or not getattr(self.calendar, "enabled", False):
            return
        if self._shift_alarm_active:
            return
        curr_dt = self.calendar.to_datetime(self.env.now)
        if self.calendar.is_working_hour(curr_dt):
            return
        gap = self.calendar.get_gap_to_next_work_window(curr_dt)
        if gap > 0:
            self._shift_alarm_active = True
            self.env.process(self._shift_alarm_worker(gap))

    def _shift_alarm_worker(self, gap):
        yield self.env.timeout(gap)
        self._shift_alarm_active = False
        self._dispatch_all_pending()

    def _dispatch_all_pending(self):
        if not self._is_working_now():
            self._ensure_shift_alarm()
            return

        i = 0
        while i < len(self.pending_requests):
            req = self.pending_requests[i]
            if req["event"].triggered:
                self.pending_requests.pop(i)
                continue

            free_workers = [w for w in req["eligible"] if self._is_worker_free(w)]
            if free_workers:
                min_idle = min(self.worker_idle_since[w] for w in free_workers)
                candidates = [w for w in free_workers if self.worker_idle_since[w] == min_idle]
                chosen = random.choice(candidates) if len(candidates) > 1 else candidates[0]

                self.pending_requests.pop(i)
                self.worker_active_count[chosen] += 1
                self.worker_busy[chosen] = not self._is_worker_free(chosen)
                self.worker_activities[chosen].append(req["activity"])
                self.worker_activity[chosen] = req["activity"]
                req["event"].succeed(value=chosen)
            else:
                i += 1

    def get_role_occupancy(self, activities_list=None):
        """Returns concurrent departmental utilization vector matching Role_*_OC."""
        if not self.role_names:
            # Fallback for legacy 1-to-1 activity pools if no roles are defined
            activities_list = activities_list or list(self.activity_pools.keys())
            oc = []
            for act in activities_list:
                eligible = self.activity_pools.get(act, self.all_workers)
                busy_on_act = sum(1 for w in eligible if act in self.worker_activities[w])
                oc.append(busy_on_act / max(1, len(eligible)))
            return oc

        oc = []
        for r in self.role_names:
            role_workers = self.role_to_workers.get(r, self.all_workers)
            active_in_role = sum(
                1 for w in role_workers
                if any(self.activity_to_role.get(act) == r for act in self.worker_activities[w])
            )
            cap = max(1, self.role_capacities.get(r, len(role_workers)))
            oc.append(min(1.0, active_in_role / cap))
        return oc

    def request_worker(self, activity):
        """
        Routes the task by organizational role and dispatches to an eligible multi-skilled
        worker using Longest-Idle-First (LIF) without creating worker ubiquity.
        """
        # Step 1: Route by Role
        role = self.activity_to_role.get(activity)
        if role and role in self.role_to_workers:
            eligible = self.role_to_workers[role]
        else:
            eligible = self.activity_pools.get(activity, self.all_workers)

        free_workers = [w for w in eligible if self._is_worker_free(w)]
        is_working = self._is_working_now()

        if is_working and free_workers:
            # Longest Idle First: select the worker who has been waiting idle the longest
            min_idle = min(self.worker_idle_since[w] for w in free_workers)
            candidates = [w for w in free_workers if self.worker_idle_since[w] == min_idle]
            chosen = random.choice(candidates) if len(candidates) > 1 else candidates[0]

            self.worker_active_count[chosen] += 1
            self.worker_busy[chosen] = not self._is_worker_free(chosen)
            self.worker_activities[chosen].append(activity)
            self.worker_activity[chosen] = activity
            ev = self.env.event()
            ev.succeed(value=chosen)
            return ev
        else:
            ev = self.env.event()
            self.pending_requests.append({
                "activity": activity,
                "role": role,
                "eligible": set(eligible),
                "event": ev,
                "req_time": self.env.now
            })
            if not is_working:
                self._ensure_shift_alarm()
            return ev

    def release_worker(self, worker_id, activity=None):
        """Releases worker; if human latency modeling is active and queue is calm, worker enters refractory buffer."""
        if activity and activity in self.worker_activities[worker_id]:
            self.worker_activities[worker_id].remove(activity)
        elif self.worker_activities[worker_id]:
            self.worker_activities[worker_id].pop()
        self.worker_activity[worker_id] = self.worker_activities[worker_id][-1] if self.worker_activities[worker_id] else None

        cooldown_enabled = self.latency_cfg.get("enabled", True) if self.latency_cfg else False
        queue_aware = self.latency_cfg.get("queue_aware", True) if self.latency_cfg else True

        # Check if any waiting task needs this worker right now
        has_pending_work = any(
            worker_id in req["eligible"] and not req["event"].triggered
            for req in self.pending_requests
        )

        # Queue-Aware Cooldown: If tickets are waiting, worker immediately takes next ticket.
        # Only take coffee break cooldown if no immediate tickets are waiting.
        if cooldown_enabled and (not queue_aware or not has_pending_work):
            cooldown = self._sample_cooldown(worker_id)
            if cooldown > 0:
                self.env.process(self._cooldown_and_dispatch(worker_id, cooldown))
                return

        self._complete_release(worker_id)

    def _sample_cooldown(self, worker_id):
        profiles = self.latency_cfg.get("per_worker_profiles", {})
        prof = profiles.get(worker_id) or self.latency_cfg
        dist = prof.get("best_distribution") or prof.get("global_distribution", "lognormal")
        params = prof.get("distribution_params") or prof.get("global_params", {})
        max_cd = float(self.latency_cfg.get("max_cooldown_seconds", 1800.0))

        try:
            if dist == "lognormal":
                val = float(st.lognorm.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "gamma":
                val = float(st.gamma.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            else:
                mean_s = float(prof.get("mean_seconds", prof.get("global_mean_seconds", 300.0)))
                val = float(random.expovariate(1.0 / max(1.0, mean_s)))
            return min(max(5.0, val), max_cd)
        except Exception:
            return float(random.expovariate(1.0 / 300.0))

    def _cooldown_and_dispatch(self, worker_id, cooldown_seconds):
        yield self.env.timeout(cooldown_seconds)
        self._complete_release(worker_id)

    def _complete_release(self, worker_id):
        self.worker_active_count[worker_id] = max(0, self.worker_active_count[worker_id] - 1)
        self.worker_busy[worker_id] = not self._is_worker_free(worker_id)
        if self._is_worker_free(worker_id):
            self.worker_idle_since[worker_id] = self.env.now

        if not self._is_working_now():
            self._ensure_shift_alarm()
            return

        self._dispatch_all_pending()


# Backward compatibility alias
WorkerManager = SharedWorkerPool


class SimulationProcess:
    """
    Simulation Orchestrator & Process Coordinator (Legacy Parity with checking_process.py).
    Manages SimPy environment, resources, calendar shifts, token lifecycles, and evaluation.
    """
    def __init__(self, cfg, num_cases=None, mode=None, arrival_mode=None, export_xes=None, pnml_path=None, duration_days=None, force_retrain_prophet=False):
        self.cfg = cfg
        seed = cfg.get("project", {}).get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        self.mode = mode or cfg.get("simulation", {}).get("mode", "hybrid_residual")
        self.num_cases = num_cases
        self.duration_days = duration_days if duration_days is not None else cfg.get("simulation", {}).get("duration_simulation")
        self.duration_seconds = float(self.duration_days * 86400.0) if (self.duration_days and float(self.duration_days) > 0) else None
        self.max_events = cfg.get("simulation", {}).get("max_events_per_case", 80)
        self.target_activities = cfg.get("target_activities", [])
        self.proc_prefixes = tuple(cfg.get("preprocessing", {}).get("process_activity_prefixes", []))
        self.ms_prefixes = tuple(cfg.get("preprocessing", {}).get("milestone_prefixes", []))
        self.force_retrain_prophet = force_retrain_prophet

        # Output paths
        self.sim_log_path = os.path.join(
            PROJECT_ROOT,
            f"data/generated/csv/simulated_log_{self.mode}.csv" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_log", "data/generated/csv/simulated_log.csv")
        )
        self.export_xes = export_xes if export_xes is not None else cfg.get("simulation", {}).get("export_xes", True)
        self.output_simulated_xes = os.path.join(
            PROJECT_ROOT,
            cfg.get("simulation", {}).get("output_simulated_xes", "data/generated/xes/simulated_log.xes")
        )

        # Arrival Mode
        arr_cfg = cfg.get("simulation", {}).get("arrival", {})
        self.arrival_mode = arrival_mode or arr_cfg.get("mode", "replay")
        self.arr_cfg = arr_cfg

        # Load Petri Net & Semantics
        target_pnml = pnml_path or os.path.join(PROJECT_ROOT, cfg.get("paths", {}).get("petri_net", "models/petri_nets/discovered_model_split.pnml"))
        if not os.path.exists(target_pnml):
            raise FileNotFoundError(f"Petri Net PNML file not found at: {target_pnml}")
        self.net, self.im, self.fm = pm4py.read_pnml(target_pnml)
        self.pnml_path = target_pnml
        self.branch_target_map = self._build_branch_target_map()

        if not self.target_activities:
            self.target_activities = sorted(list(set(
                t.label for t in self.net.transitions if t.label is not None
            )))
            print(f"Target activities: [Auto-discovered {len(self.target_activities)} activities from Petri Net]: {self.target_activities}")

        # Calendar Manager
        self.calendar = CalendarManager.from_config(cfg)

        # Load Causal Delays
        models_dir_rel = cfg.get("paths", {}).get("models_dir", "models")
        delay_manifest_path = os.path.join(PROJECT_ROOT, models_dir_rel, "delays/delay_manifest.json")
        self.delay_manifest = None
        self.external_delays = {}
        self.ex_ante_delays = {}
        if os.path.exists(delay_manifest_path):
            try:
                with open(delay_manifest_path, "r") as f:
                    self.delay_manifest = json.load(f)

                # Check configuration to include internal queue delays if requested
                delays_cfg = cfg.get("simulation", {}).get("extraneous_delays", {})
                delays_enabled = delays_cfg.get("enabled", True)
                if delays_enabled:
                    use_all = delays_cfg.get("include_internal_queue_delays", False)
                    active_delays = list(self.delay_manifest.get("external_uncoupled_delays", []))
                    if use_all:
                        active_delays.extend(self.delay_manifest.get("internal_queue_delays", []))

                    for d in active_delays:
                        # Ex-Ante activity-level delays (Chapela-Campa)
                        act = d.get("activity")
                        if act and d.get("placement", "ex_ante") == "ex_ante":
                            self.ex_ante_delays[act] = d

                        # Arc-level delays
                        trig = d.get("trigger_activity")
                        res = d.get("resume_activity")
                        if trig and res:
                            self.external_delays[(trig, res)] = d

                    # Also register causal transition arcs if present
                    for d in self.delay_manifest.get("causal_transition_arcs", []):
                        self.external_delays[(d["trigger_activity"], d["resume_activity"])] = d

            except Exception as e:
                print(f"[!] Warning: Failed to load delay manifest: {e}")

        # SimPy Environment & Worker Pool
        self.env = simpy.Environment()
        role_mapping_path = os.path.join(PROJECT_ROOT, models_dir_rel, "role_mapping.json")
        latency_cfg = dict(self.delay_manifest.get("human_inter_ticket_latency", {})) if self.delay_manifest else {}
        user_hl = cfg.get("simulation", {}).get("human_latency", {})
        if isinstance(user_hl, dict):
            latency_cfg.update(user_hl)
        self.workers = SharedWorkerPool(self.env, role_mapping_path, latency_cfg=latency_cfg, calendar=self.calendar)

        # Runtime Champion Predictor
        self.predictor = ChampionPredictor(cfg)

        # Arrival Generator
        self.arrivals_generator = ArrivalGenerator(
            cfg, num_cases=self.num_cases, arrival_mode=self.arrival_mode, calendar=self.calendar, duration_days=self.duration_days,
            force_retrain_prophet=self.force_retrain_prophet
        )

        # Evaluator
        self.evaluator = SimulationEvaluator(
            cfg, mode=self.mode, arrival_mode=self.arrival_mode, export_xes=self.export_xes,
            sim_log_path=self.sim_log_path, output_simulated_xes=self.output_simulated_xes,
            workers_count=len(self.workers.all_workers), calendar=self.calendar
        )

        # Dynamic State Tracking
        self.simulated_records = []
        self.active_cases_count = 0
        self.ac_wip = defaultdict(int)

    def _build_branch_target_map(self):
        """
        Traces Petri net arcs and silent transitions downstream to map
        (decision_place, transition_name) -> target observable activity or '<END>'.
        """
        trans_map = {t.name: t.label for t in self.net.transitions}
        is_silent = {t.name: (t.label is None) for t in self.net.transitions}
        out_arcs = defaultdict(list)
        for a in self.net.arcs:
            out_arcs[a.source.name].append(a.target.name)

        def resolve_target(t_name, visited=None):
            if visited is None:
                visited = set()
            if t_name in visited:
                return None
            visited.add(t_name)
            if not is_silent.get(t_name, False):
                return trans_map.get(t_name, t_name)
            for p in out_arcs.get(t_name, []):
                if p == "sink":
                    return "<END>"
                for dt in out_arcs.get(p, []):
                    if not is_silent.get(dt, False):
                        return trans_map.get(dt, dt)
                    res = resolve_target(dt, visited.copy())
                    if res:
                        return res
            return trans_map.get(t_name, t_name)

        branch_map = {}
        for p in sorted(self.net.places, key=lambda pl: pl.name):
            if len(p.out_arcs) >= 2:
                for arc in sorted(p.out_arcs, key=lambda a: a.target.name):
                    t = arc.target
                    branch_map[(p.name, t.name)] = resolve_target(t.name)
        return branch_map

    def has_ex_ante_delay(self, activity):
        """Returns True if activity has a discovered Chapela-Campa Ex-Ante extraneous delay."""
        return activity in self.ex_ante_delays

    def sample_ex_ante_delay(self, activity):
        """Samples from empirical causal distribution for Ex-Ante extraneous delay."""
        d = self.ex_ante_delays.get(activity)
        if not d:
            return 0.0
        return self._sample_delay_dist(d)

    def sample_external_delay(self, trigger_act, resume_act):
        """Samples from empirical causal distribution for external uncoupled delay."""
        d = self.external_delays.get((trigger_act, resume_act))
        if not d:
            return 0.0
        return self._sample_delay_dist(d)

    def _sample_delay_dist(self, d):
        dist = d.get("best_distribution", "lognormal")
        params = d.get("distribution_params", {})
        try:
            if dist == "gamma":
                val = float(st.gamma.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "exponential":
                val = float(st.expon.rvs(loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "lognormal":
                val = float(st.lognorm.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            else:
                val = float(d.get("median_hours", 0.0) * 3600.0)
            return max(0.0, val)
        except Exception:
            return max(0.0, float(d.get("median_hours", 0.0) * 3600.0))

    def record_event(self, record_dict):
        """Appends an event execution record to the simulation trace buffer."""
        self.simulated_records.append(record_dict)

    def run(self):
        """Executes the discrete-event simulation engine and returns evaluation metrics."""
        print("\n" + "=" * 80)
        print("      RIMS+ MODERN DISCRETE-EVENT SIMULATOR (SIMPY)")
        print(f"      Mode:        [{self.mode.upper()}]")
        print(f"      Arrivals:    [{self.arrival_mode.upper()}] Mode")
        print(f"      Petri Net:   {os.path.basename(self.pnml_path)}")
        if self.workers.role_names:
            print(f"      Roles:       {len(self.workers.role_names)} Discovered Departments ({', '.join(self.workers.role_names)})")
        print(f"      Workers:     {len(self.workers.all_workers)} Distinct Multi-Skilled Human Agents")
        print(f"      Calendar:    Working Hours {self.calendar.hour_start}:00 - {self.calendar.hour_end}:00 (Mon-Fri)")
        if self.calendar.role_calendars:
            print(f"      Per-Role:    {len(self.calendar.role_calendars)} Custom Role Schedules Active")
        ext_count = len(self.external_delays)
        print(f"      Delays:      {ext_count} Causal External Arcs Active")
        hl_cfg = self.delay_manifest.get("human_inter_ticket_latency", {}) if self.delay_manifest else {}
        hl_str = f"Active ({hl_cfg.get('global_mean_seconds', 325.4)}s mean gap)" if hl_cfg.get("enabled") else "Disabled"
        print(f"      Latency:     Human Inter-Ticket Latency {hl_str}")
        if self.duration_seconds is not None:
            print(f"      Duration:    {self.duration_days} days ({self.duration_seconds:,.0f}s SimPy clock limit)")
        print("=" * 80)

        # Prepare Case Arrivals
        arrivals, df_real = self.arrivals_generator.prepare_arrivals()
        print(f"Scheduled {len(arrivals):,} cases for simulation.")

        # Reference start datetime
        ref_dt = arrivals[0]["arrival_time"]
        self.calendar.set_ref_dt(ref_dt)

        # Launch SimPy Arrival Processes via Token instances
        for arr in arrivals:
            offset_seconds = (arr["arrival_time"] - ref_dt).total_seconds()
            token = Token(
                case_id=arr["case_id"],
                arrival_offset=offset_seconds,
                requested_amount=arr.get("requested_amount", 0.0),
                process=self,
                init_ms=arr.get("init_ms"),
                ms_states=arr.get("ms_states"),
                inter_arrival_seconds=arr.get("inter_arrival_seconds", 0.0),
                case_attrs=arr.get("case_attrs", {})
            )
            self.env.process(token.run())

        start_exec = time.time()
        print("\nExecuting simulation engine...")
        if self.duration_seconds is not None:
            self.env.run(until=self.duration_seconds)
        else:
            self.env.run()
        elapsed = time.time() - start_exec
        print(f"\n[✓] Simulation completed in {elapsed:.2f} seconds ({len(self.simulated_records):,} events generated).")

        # Save simulated event log (CSV)
        df_sim = pd.DataFrame(self.simulated_records)
        self.evaluator.save_csv(df_sim)

        # Save simulated event log (Standard IEEE XES)
        self.evaluator.save_xes(df_sim)

        # Run Validation & Benchmark Evaluation
        return self.evaluator.evaluate(df_real, df_sim, elapsed)


# Backward compatibility alias
DiscreteEventSimulation = SimulationProcess