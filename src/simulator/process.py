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
    def __init__(self, env, role_mapping_path, latency_cfg=None):
        self.env = env
        self.latency_cfg = latency_cfg or {}
        with open(role_mapping_path, "r") as f:
            data = json.load(f)

        self.all_workers = data.get("__all_resources__", [])
        if not self.all_workers:
            self.all_workers = sorted(list(set.union(*[set(data[k]) for k in data if not k.startswith("__")])))

        self.activity_pools = {k: data[k] for k in data if not k.startswith("__")}
        self.worker_busy = {w: False for w in self.all_workers}
        self.worker_activity = {w: None for w in self.all_workers}
        self.pending_requests = []

    def get_role_occupancy(self, activities_list):
        """Returns concurrent activity utilization vector matching Role_*_OC."""
        oc = []
        for act in activities_list:
            eligible = self.activity_pools.get(act, self.all_workers)
            busy_on_act = sum(1 for w in eligible if self.worker_busy[w] and self.worker_activity[w] == act)
            oc.append(busy_on_act / max(1, len(eligible)))
        return oc

    def request_worker(self, activity):
        """Requests an available worker from the activity's eligible skill pool."""
        eligible = self.activity_pools.get(activity, self.all_workers)
        free_workers = [w for w in eligible if not self.worker_busy[w]]

        if free_workers:
            chosen = random.choice(free_workers)
            self.worker_busy[chosen] = True
            self.worker_activity[chosen] = activity
            ev = self.env.event()
            ev.succeed(value=chosen)
            return ev
        else:
            ev = self.env.event()
            self.pending_requests.append({
                "activity": activity,
                "eligible": set(eligible),
                "event": ev,
                "req_time": self.env.now
            })
            return ev

    def release_worker(self, worker_id):
        """Releases worker; if human latency modeling is active, worker enters refractory buffer."""
        self.worker_activity[worker_id] = None
        if self.latency_cfg and self.latency_cfg.get("enabled", True):
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

        try:
            if dist == "lognormal":
                val = float(st.lognorm.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            elif dist == "gamma":
                val = float(st.gamma.rvs(params["shape"], loc=params.get("loc", 0.0), scale=params["scale"]))
            else:
                mean_s = float(prof.get("mean_seconds", prof.get("global_mean_seconds", 300.0)))
                val = float(random.expovariate(1.0 / max(1.0, mean_s)))
            return min(max(5.0, val), 1800.0)  # Clamp between 5s and 30 mins
        except Exception:
            return float(random.expovariate(1.0 / 300.0))

    def _cooldown_and_dispatch(self, worker_id, cooldown_seconds):
        yield self.env.timeout(cooldown_seconds)
        self._complete_release(worker_id)

    def _complete_release(self, worker_id):
        self.worker_busy[worker_id] = False
        for i, req in enumerate(self.pending_requests):
            if worker_id in req["eligible"] and not req["event"].triggered:
                self.pending_requests.pop(i)
                self.worker_busy[worker_id] = True
                self.worker_activity[worker_id] = req["activity"]
                req["event"].succeed(value=worker_id)
                return


# Backward compatibility alias
WorkerManager = SharedWorkerPool


class SimulationProcess:
    """
    Simulation Orchestrator & Process Coordinator (Legacy Parity with checking_process.py).
    Manages SimPy environment, resources, calendar shifts, token lifecycles, and evaluation.
    """
    def __init__(self, cfg, num_cases=None, mode=None, arrival_mode=None, export_xes=None, pnml_path=None):
        self.cfg = cfg
        self.mode = mode or cfg.get("simulation", {}).get("mode", "hybrid_residual")
        self.num_cases = num_cases
        self.max_events = cfg.get("simulation", {}).get("max_events_per_case", 80)
        self.target_activities = cfg.get("target_activities", [])
        self.ms_prefixes = tuple(cfg.get("preprocessing", {}).get("milestone_prefixes", []))

        # Output paths
        self.sim_log_path = os.path.join(
            PROJECT_ROOT,
            f"data/processed/simulated_log_{self.mode}.csv" if self.mode != "hybrid_residual" else
            cfg.get("simulation", {}).get("output_simulated_log", "data/processed/simulated_log.csv")
        )
        self.export_xes = export_xes if export_xes is not None else cfg.get("simulation", {}).get("export_xes", True)
        self.output_simulated_xes = os.path.join(
            PROJECT_ROOT,
            cfg.get("simulation", {}).get("output_simulated_xes", "data/processed/simulated_log.xes")
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

        # Calendar Manager
        cal_cfg = cfg.get("simulation", {}).get("calendar", {})
        self.calendar = CalendarManager(
            enabled=cal_cfg.get("enabled", True),
            work_days=cal_cfg.get("work_days", [0, 1, 2, 3, 4]),
            hour_start=cal_cfg.get("hour_start", 8),
            hour_end=cal_cfg.get("hour_end", 17),
            per_role=cal_cfg.get("per_role", {})
        )

        # Load Causal Delays
        delay_manifest_path = os.path.join(PROJECT_ROOT, "models/delays/delay_manifest.json")
        self.delay_manifest = None
        self.external_delays = {}
        if os.path.exists(delay_manifest_path):
            try:
                with open(delay_manifest_path, "r") as f:
                    self.delay_manifest = json.load(f)
                for d in self.delay_manifest.get("external_uncoupled_delays", []):
                    self.external_delays[(d["trigger_activity"], d["resume_activity"])] = d
            except Exception as e:
                print(f"[!] Warning: Failed to load delay manifest: {e}")

        # SimPy Environment & Worker Pool
        self.env = simpy.Environment()
        role_mapping_path = os.path.join(PROJECT_ROOT, "models/role_mapping.json")
        latency_cfg = self.delay_manifest.get("human_inter_ticket_latency") if self.delay_manifest else None
        self.workers = SharedWorkerPool(self.env, role_mapping_path, latency_cfg=latency_cfg)

        # Runtime Champion Predictor
        self.predictor = ChampionPredictor(cfg)

        # Arrival Generator
        self.arrivals_generator = ArrivalGenerator(
            cfg, num_cases=self.num_cases, arrival_mode=self.arrival_mode, calendar=self.calendar
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
        for p in self.net.places:
            if len(p.out_arcs) >= 2:
                for arc in p.out_arcs:
                    t = arc.target
                    branch_map[(p.name, t.name)] = resolve_target(t.name)
        return branch_map

    def sample_external_delay(self, trigger_act, resume_act):
        """Samples from empirical causal distribution for external uncoupled delay."""
        d = self.external_delays.get((trigger_act, resume_act))
        if not d:
            return 0.0
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
        print(f"      Workers:     {len(self.workers.all_workers)} Distinct Multi-Skilled Human Agents")
        print(f"      Calendar:    Working Hours {self.calendar.hour_start}:00 - {self.calendar.hour_end}:00 (Mon-Fri)")
        if self.calendar.role_calendars:
            print(f"      Per-Role:    {len(self.calendar.role_calendars)} Custom Role Schedules Active")
        ext_count = len(self.external_delays)
        print(f"      Delays:      {ext_count} Causal External Arcs Active")
        hl_cfg = self.delay_manifest.get("human_inter_ticket_latency", {}) if self.delay_manifest else {}
        hl_str = f"Active ({hl_cfg.get('global_mean_seconds', 325.4)}s mean gap)" if hl_cfg.get("enabled") else "Disabled"
        print(f"      Latency:     Human Inter-Ticket Latency {hl_str}")
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
                requested_amount=arr["requested_amount"],
                process=self,
                init_ms=arr["init_ms"],
                ms_states=arr["ms_states"]
            )
            self.env.process(token.run())

        start_exec = time.time()
        print("\nExecuting simulation engine...")
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