# RIMS++: Runtime Integration of Machine Learning & Simulation

**RIMS++** is a Business Process Simulation (BPS) framework that combines Petri net process models with Machine Learning. It discovers process structure, organizational roles, external delays, and routing rules directly from event logs, and simulates process executions in SimPy.

What the framework does:
* **Discovers and Executes Petri Nets**: Mines workflow nets from the training log (via Split Miner) and checks soundness and executes them using token marking semantics in SimPy.
* **Predicts Durations & Waiting Times with ML**: Trains machine learning models (XGBoost, TCN, LSTM) to predict activity processing and waiting times based on real-time process state (WIP, time of day, queue congestion).
* **Mines Organizational Roles & Manages Workers**: Discovers roles from the log (*Burattin, Sperduti, & Veluscek, 2013*) and simulates named workers across departments, tracking individual availability, multi-tasking limits, and enforcing shared capacity limits across their assignments.
* **Identifies External Delays & Cooldowns**: Detects delays caused by external factors (e.g. waiting for a customer response) or worker post-task cooldowns using causal estimators (*Chapela-Campa & Dumas, 2022, 2024*).
* **Learns Decision-Based Routing**: Trains white-box decision trees at XOR split places to route cases based on data attributes and past trace history, falling back to probabilities when needed.
* **Models Case Arrivals**: Generates new cases through historical replay, statistical distributions, or time-series forecasting with Facebook Prophet.

---

## Repository Structure

```
fbk_rims/
├── run_pipeline.py                    # Master pipeline orchestrator (runs steps 00 through 06)
├── config.yaml                        # Central configuration (paths, miner, roles, features, models, simulation)
├── requirements.txt                   # Formal project dependencies (PyTorch, SimPy, PM4Py, Prophet, XGBoost)
├── context.md                         # Detailed project context, evolution, and architectural blueprint
├── README.md                          # Primary reference documentation
├── docs/                              # Flowcharts and architecture documentation
│   ├── causal_discovery_flowchart.md  # Causal test flowchart for extraneous delays
│   ├── code_audit_2026-11-06.md        # Audit findings, including remaining limitations
│   └── first16_fixes.md               # Implemented fixes, semantics and rebuild instructions
├── data/                              # Data directories
│   ├── raw/                           # Raw input event logs (.xes / .csv)
│   ├── processed/                     # Cleaned logs, training tables, and benchmark reports
│   │   ├── <log_name>_filtered.xes    # Preprocessed and standardized event log
│   │   ├── aligned_<log_name>.xes     # Alignment-repaired log, including synthetic model moves
│   │   ├── datasets/                  # Feature-engineered tabular datasets
│   │   │   ├── dataset_duration_global.csv     # Duration dataset (0-duration filtered)
│   │   │   ├── dataset_waiting_time_global.csv # Waiting time dataset (0-waiting preserved)
│   │   │   └── activity_weights.json           # Smoothed inverse sample importance weights
│   │   ├── case_split.json            # Shared source-case boundaries and train/validation/test assignments
│   │   ├── simulation_metrics.json    # Metrics, populations, units and error statuses
│   │   ├── simulation_report.md       # Latest simulation run benchmark report
│   │   └── ablation_report.md         # Simulation ablation study leaderboard
│   └── generated/                     # Ephemeral simulation outputs (ignored by Git)
│       ├── csv/simulated_log.csv      # Events; unfinished executions have no completion timestamp
│       ├── csv/simulated_log_cases.csv # Case completion/censoring/failure statuses
│       └── xes/simulated_log.xes      # Output simulated event log (standard IEEE XES)
├── models/                            # Trained model artifacts and manifests
│   ├── activity_profiles.json         # Data-driven activity behavior profiles (Human vs. Automated vs. Instant)
│   ├── activity_profiles_report.md    # Activity profiling statistical breakdown report
│   ├── role_mapping.json              # Discovered organizational role partition, rosters, and capacities
│   ├── role_discovery_report.md       # Organizational role discovery markdown report
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   └── discovered_model_split.pnml     # Workflow net discovered via Split Miner
│   ├── delays/                        # Causal delays and worker refractory latencies
│   │   ├── delay_manifest.json        # Machine-readable delay manifest with KS-fitted distributions
│   │   └── delay_discovery_report.md  # Statistical delay discovery report
│   ├── duration/                      # Activity duration models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Validation-selected dispatch and separate candidate bundles
│   │   ├── global_{xgboost|tcn|lstm}/  # Global model checkpoints & scalers
│   │   ├── local_{xgboost|tcn|lstm}/   # Activity-specialized local model checkpoints & scalers
│   │   └── benchmark_report.md        # Validation coverage and separate champion test results
│   ├── waiting_time/                  # Activity waiting time models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Waiting time champion dispatch & scalers
│   │   ├── global_{xgboost|tcn|lstm}/  # Global model checkpoints & scalers
│   │   ├── local_{xgboost|tcn|lstm}/   # Activity-specialized local model checkpoints & scalers
│   │   └── benchmark_report.md        # Waiting time benchmark leaderboard
│   ├── routing/                       # White-box XOR decision classifiers & routing manifest
│   │   ├── routing_decisions.json     # Dynamic decision dispatch manifest
│   │   └── routing_*.joblib           # Trained decision tree / random forest classifiers per XOR place
│   └── prophet/                       # Serialized Facebook Prophet arrival models & metadata
│       ├── prophet_arrivals_*.json    # Serialized model checkpoint for sub-second startup
│       └── prophet_arrivals_*_meta.json # Training metadata, log timestamp, and carrying capacity
├── src/                               # Active RIMS++ Core Pipeline & Simulator Package
│   ├── 00_filter_log.py               # Step 0: Raw event log preprocessing, cleaning & standardization
│   ├── 01_process_discovery.py        # Step 1: Petri net discovery (Split Miner)
│   ├── 01b_discover_roles.py          # Step 1b: Organizational role mining
│   ├── view_models.py                 # Step 1c: Petri net visualization (.png export)
│   ├── 02_align_and_repair_log.py     # Step 2: Log alignment & replay repair (PM4Py A* state space)
│   ├── 02b_discover_delays.py         # Step 2b: Causal external delay & worker latency discovery
│   ├── historical_features.py         # Historical event sweep and prediction-time snapshots
│   ├── temporal_split.py              # Shared case split, training filters and artifact provenance
│   ├── model_artifacts.py             # Experiment identity and valid-prediction scoring
│   ├── 03_dataset_builder.py          # Step 3: Feature engineering & state vector extraction
│   ├── activity_profiler.py           # Activity behavioral profiling (Human vs. Automated vs. Gateway)
│   ├── features.py                    # Declarative Feature Registry & context builder
│   ├── 04_train.py                    # Step 4: Multi-architecture training engine (XGB, TCN, LSTM)
│   ├── 04b_benchmark_models.py        # Step 4b: Automated benchmarking & champion synthesis
│   ├── 05_train_routing.py            # Step 5: White-Box XOR decision mining & Hyperopt tuning
│   ├── 06_simulate.py                 # Step 6: Simulation CLI runner
│   ├── simulator/                     # Modular OOP Simulation Package
│   │   ├── __init__.py                # Package exports
│   │   ├── calendar.py                # CalendarManager (working shifts, night/weekend rollover)
│   │   ├── process.py                 # SimulationProcess & SharedWorkerPool
│   │   ├── token.py                   # Token (PM4Py Petri net marking semantics, XOR branching)
│   │   ├── predictor.py               # ChampionPredictor (Runtime XGBoost, TCN, LSTM inference)
│   │   ├── arrivals.py                # ArrivalGenerator (Replay, Generative, Prophet time-series)
│   │   └── metrics.py                 # SimulationEvaluator (IEEE XES/CSV export, Wasserstein, RED/EMD)
│   └── config_loader.py               # Central config loader helper
├── RIMS/                              # Cloned Reference / Legacy Repo (baseline, NOT the simulator)
└── RIMS_tool/                         # Cloned Reference / Legacy Repo (BPM 2023 demo tool, NOT the simulator)
```

---

## Shared helper modules

These modules are imported by the existing pipeline stages; they are not additional commands or pipeline steps:

| Module | Responsibility | Main callers |
|---|---|---|
| `historical_features.py` | Pair observed executions, maintain historical case/activity/worker state, and capture duration-at-START and waiting-at-eligibility snapshots. | Step 03 dataset builder |
| `temporal_split.py` | Create/reuse one case split, filter fitting data to training cases, and check source-split/artifact provenance. | Discovery, profiling, dataset building, training and arrival fitting |
| `model_artifacts.py` | Identify compatible training experiments and calculate scores only for available predictions. | Step 04 training and Step 04b champion synthesis |

They keep shared rules in one place rather than duplicating them across scripts. `features.py` still defines the feature extractors; `historical_features.py` supplies their historical inputs.

---

## Setup & Prerequisites

Activate your Python virtual environment (Python 3.10+ recommended) and install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies:
* `pm4py` : Process discovery, conformance checking, and state-space alignments.
* `simpy` : Asynchronous discrete-event simulation engine.
* `torch` : Deep Learning neural network architectures (**TCN** and **LSTM**).
* `xgboost` : Gradient-boosted decision tree regression and classification.
* `scikit-learn` : Decision trees, random forests, feature scalers, and validation metrics.
* `hyperopt` : Bayesian Tree-structured Parzen Estimator (TPE) hyperparameter optimization.
* `prophet` : Time-series forecasting for non-linear case arrival patterns.
* `pandas`, `numpy`, `scipy` : Tabular manipulation, state extraction, and Kolmogorov–Smirnov distribution fitting.
* `pyyaml` : Centralized configuration management.

---

## Quick Start: Master Pipeline Runner (`run_pipeline.py`)

RIMS++ provides an all-in-one orchestrator that reads [`config.yaml`](config.yaml) and executes the entire pipeline end-to-end with a single command:

```bash
# 1. Run the entire pipeline end-to-end (Steps 00 through 06)
python run_pipeline.py

# 2. Run with a custom configuration file
python run_pipeline.py --config config.yaml

# 3. Run specific stages only (e.g. skip discovery, train models & simulate)
python run_pipeline.py --steps 3,4,4b,5,6

# 4. Fast simulation run overriding case count
python run_pipeline.py --steps 06 --cases 100

# 5. Dry-run preview of planned stages
python run_pipeline.py --dry-run
```

---

## Central Configuration (`config.yaml`)

The entire operational pipeline is governed declaratively through [`config.yaml`](config.yaml).

### 1. General Paths & Project Metadata
```yaml
project:
  name: "My_Process"
  random_seed: 42

paths:
  raw_log: "data/processed/<log_name>_filtered.xes"
  aligned_log: "data/processed/aligned_<log_name>.xes"
  petri_net: "models/petri_nets/discovered_model_split.pnml"
  output_base_dir: "data/processed"
  models_dir: "models"
  routing_models_dir: "models/routing"
```

### 2. Process Discovery Settings
Configures the Split Miner algorithm and graph simplification thresholds:
```yaml
discovery:
  split_miner:
    epsilon: 0.1
    eta: 0.4
    minimize_or_joins: true
    variant: "classic"
```

### 3. Organizational Roles & Resource Allocation
Configures the Burattin–Sperduti role mining algorithm and runtime worker dispatching:
```yaml
resources:
  discovery:
    algorithm: "sperduti"           # Burattin–Sperduti role mining algorithm
    threshold: 0.35                 # Jaccard clustering threshold (0.35 -> macro departments; >= 0.60 -> 1 role per activity)
    output_file: "models/role_mapping.json"
    report_file: "models/role_discovery_report.md"
  simulation:
    mode: "hybrid_sperduti"         # Route by role, dispatch to multi-skilled named workers
    dispatch_policy: "longest_idle_first"
    enable_concurrency_quotas: true # Enables empirical multi-tasking quotas (C_w >= 1) mined from log
```

### 4. Activity Behavioral Profiling
Configures automatic classification of activities into operational archetypes based on empirical duration and queue characteristics:
```yaml
activity_profiling:
  enabled: true
  zero_duration_threshold: 0.95   # Fraction of zero-duration instances to qualify as automated/instant
  min_automated_wait_seconds: 60.0 # Waiting time boundary distinguishing automated tasks from instant gateways
  output_file: "models/activity_profiles.json"
  report_file: "models/activity_profiles_report.md"
  overrides: {}                   # Optional explicit manual overrides {"Activity_Name": "human_work_item" | "automated_task" | "instant_gateway"}
```

### 5. Feature Engineering Registry
Controls dynamic state features extracted chronologically from the event log without future data leakage:
```yaml
features:
  prefix_window_size: 10          # History window of past activities (trace prefix)
  selected_features:
    - wip                         # Process-level active case count
    - ac_wip                      # Activity-level active case count
    - daytime                     # Normalized time-of-day in [0, 1]
    - weekday                     # One-hot weekday indicators Weekday_0..6
    - role_occupancy              # Real-time departmental utilization vector (Role_*_OC)
    - prev_duration               # Processing duration of immediately preceding activity
    - trace_history               # Cumulative past activity presence indicators (has_*)
    - case_attributes             # Domain-specific case payload attributes
    # - resource_id               # Worker identity (optional)

  # Cumulative Trace History Settings
  trace_history:
    mode: "profile_aware"         # Options: "profile_aware" (auto INSTANT_GATEWAY/AUTOMATED_TASK), "all_activities", "whitelist"
    include_counts: false         # true = count_* frequency counts, false = binary has_* only
    whitelist: []                 # Specific activity names to track if mode is "whitelist"

  # Declarative Case Attributes
  case_attributes:
    - column: "case:AMOUNT_REQ"
      feature_name: "RequestedAmount"
      type: "continuous"
      treat_zero_as_missing: true
      fill_strategy: "median"
      fill_value: 0.0

  event_attributes: []
```

### 6. Log Preprocessing & Standardization
Handles activity standardization, lifecycle pairing, and attribute mapping:
```yaml
preprocessing:
  input_log: "data/raw/<raw_event_log>.xes"

  # Optional Column Mapping: maps custom column names in the raw log to internal standard attributes
  # Defaults match standard IEEE XES names. Change if using custom CSV/XES logs.
  columns:
    case_id: "case:concept:name"
    activity: "concept:name"
    timestamp: "time:timestamp"
    resource: "org:resource"
    lifecycle: "lifecycle:transition"

  # Optional Lifecycle Transition Value Mapping: maps custom values to standard START / COMPLETE
  lifecycle_mapping:
    start: ["START", "start", "begin", "open"]
    complete: ["COMPLETE", "complete", "finish", "end", "close"]

  # Optional activity renaming dictionary (e.g. translation or normalization):
  activity_mapping: {}

# Activities to model (Duration, Waiting Time, Routing)
# If empty list [] or omitted, RIMS++ automatically discovers and models all activities in the log.
target_activities: []
```

> [!NOTE]
> **Universal Log-Agnostic Modeling & Automated Task Discovery**:
> RIMS++ is 100% event log agnostic and models the entire operational process in a unified Petri net without artificial prefix carving or manual filtering rules.
> Rather than discarding instant status events or hardcoding milestone prefixes (e.g. `W_`, `A_`, `O_`), the data-driven **`ActivityProfiler`** analyzes historical execution durations and waiting times across all activities, categorizing them into:
> 1. **`HUMAN_WORK_ITEM`**: Operations with measurable processing duration requiring human workforce capacity and constrained by business operating shifts.
> 2. **`AUTOMATED_TASK`**: Automated system operations with $0.0\text{s}$ duration that run 24/7 without consuming human staff capacity, but may exhibit external or batching queue delays.
> 3. **`INSTANT_GATEWAY`**: Pure routing/decision markers or status milestones with $0.0\text{s}$ duration and negligible delay ($\le 60\text{s}$) that advance the Petri net marking instantaneously.

### 7. Config-Driven Task Specifications
```yaml
tasks:
  duration:
    target_column: "Duration_Seconds"
    output_file: "datasets/dataset_duration_global.csv"
    filter_zero_duration: true    # Exclude 0.0s automated/pass-through transitions
    sample_weighting:
      enabled: true
      smoothing: "sqrt"           # Options: "sqrt", "linear", "none"
      max_weight_cap: 8.0         # Gradient ceiling for rare activities
      output_weights_file: "datasets/activity_weights.json"

  waiting_time:
    target_column: "Wait_Time_Seconds"
    output_file: "datasets/dataset_waiting_time_global.csv"
    filter_zero_duration: false   # 0.0s waiting time is physically valid (immediate pickup from queue)
    sample_weighting:
      enabled: false

  routing:
    classifier_type: "decision_tree" # Options: "decision_tree" (white-box), "random_forest", "xgboost"
    mode: "payload"                  # Options: "payload", "inter_arrival", "both"
    max_depth: 21                    # Depth search ceiling for Hyperopt Bayesian search
    min_samples: 30                  # Minimum observations at XOR place required to train an ML model
    min_f1_threshold: 0.60           # If Macro F1 < 0.60, falls back to empirical branching probabilities
    output_dir: "models/routing"
```

### 8. Simulation Strategy & Environment
```yaml
simulation:
  mode: "hybrid_residual"         # Options: "hybrid_residual" (ML + physics) or "pure_physics"
  max_silent_transitions: 500     # Abort with a diagnostic on excessive silent firing
  duration_simulation: null      # Optional hard cutoff in days; unfinished cases are censored
  output_simulated_log: "data/generated/csv/simulated_log.csv"
  output_simulated_xes: "data/generated/xes/simulated_log.xes"
  extraneous_delays:
    enabled: true
    estimator: "eclaw_plus"       # Chapela-Campa & Dumas estimators: "eclaw_plus", "eclaw", "naive"
    placement: "ex_ante"
  human_latency:
    enabled: true                 # Simulate post-task refractory periods (coffee breaks)
    queue_aware: true             # Queue-aware: only take break when queue has no pending tickets
    max_cooldown_seconds: 1800.0  # Max cooldown ceiling
  arrival:
    mode: "prophet"               # Options: "prophet", "replay", "generative"
    prophet:
      growth: "logistic"
      cap_multiplier: 1.25
      sampling_method: "triangular"
      daily_seasonality: true
      weekly_seasonality: true
      max_search_days: 3650       # Count-only forecast search bound; report unmet case counts
      model_cache_dir: "models/prophet"
  calendar:
    enabled: true                 # If true, shifts work to business hours & pauses weekends
    work_days: [0, 1, 2, 3, 4]    # Monday to Friday
    hour_start: 8                 # 08:00
    hour_end: 20                  # 20:00
```

---

## End-to-End Operational Pipeline

All pipeline steps can be executed sequentially from the project root with the virtual environment active:

```bash
source venv/bin/activate
```

---

### Step 0: Log Preprocessing & Standardization (`src/00_filter_log.py`)

Preprocesses raw event logs, standardizes activity names via configurable mappings, and handles lifecycle transitions (`start` / `complete` event pairing to calculate processing duration and waiting times):

```bash
python src/00_filter_log.py
```
* **Input**: `data/raw/<raw_event_log>.xes`
* **Output**:
  * `data/processed/<log_name>_filtered.xes` (clean operational event log)

---

### Step 1: Process Discovery (`src/01_process_discovery.py`)

Discovers a structured workflow Petri net from the preprocessed log and verifies mathematical soundness (no deadlocks, proper termination) via PM4Py/Woflan checks:

```bash
python src/01_process_discovery.py
```
* **Input**: `data/processed/<log_name>_filtered.xes`
* **Output**: `models/petri_nets/discovered_model_split.pnml`

To render high-resolution visual diagrams (`.png`) of discovered Petri nets:
```bash
python src/view_models.py
```
* **Outputs**: `models/petri_nets/*_visual.png`

---

### Step 1b: Organizational Role Discovery (`src/01b_discover_roles.py`)

Discovers the underlying organizational structure and operational departments directly from the event log using the **Burattin–Sperduti Multiset Jaccard Similarity Algorithm** (*Burattin, Sperduti, & Veluscek, 2013*):

1. **Multiset Jaccard Similarity**: Evaluates pairwise activity profile overlap between all recorded human originators:

$$
S(a_i, a_j) = \frac{\sum_{r \in R} \min\big(p_i(r), p_j(r)\big)}{\sum_{r \in R} \max\big(p_i(r), p_j(r)\big)}
$$

   where $p_i(r)$ denotes the number of times worker $r$ executed activity $a_i$.

2. **Agglomerative Department Partitioning**:
   * Controlled by the similarity threshold $\theta$ (`config.yaml: resources.discovery.threshold`):
     * A moderate threshold (e.g. $\theta = 0.35$) groups activities with shared personnel into macro functional departments.
     * A strict threshold (e.g. $\theta \ge 0.60$) separates tasks into highly specialized roles (approaching 1 role per activity).
   * Maps every activity to a discovered role, lists all qualified human workers, and calculates effective staffing capacity.

3. **Empirical Multi-Tasking Quotas ($C_w$)**:
   Computes the 95th percentile of concurrent active cases per worker directly from historical event intervals:

$$
C_w = \text{Percentile}_{95}\big(\text{ActiveCases}(w, t)\big)
$$

   This enables workers to interleave tasks naturally (e.g. desk office work, phone follow-ups) without causing artificial queue serialization.

```bash
# Discover organizational roles using configured similarity threshold θ
python src/01b_discover_roles.py --threshold 0.35
```
* **Inputs**: `data/processed/aligned_<log_name>.xes` (or `<log_name>_filtered.xes`)
* **Outputs**: `models/role_mapping.json` and `models/role_discovery_report.md`

> [!TIP]
> **Anonymous Staffing Fallback**: If an input event log completely lacks an `org:resource` column (or all resources are blank/unrecorded), `01b_discover_roles.py` automatically synthesizes an unconstrained enterprise staffing pool (`Role_0_General_Operations` with `Worker_Generic`), ensuring downstream training and simulation proceed smoothly without manual intervention.

---

### Step 1c: Activity Behavioral Profiling (`src/activity_profiler.py`)

Automatically classifies every activity in the log into behavioral archetypes based on empirical duration and waiting times, eliminating hardcoded activity prefixes and enabling log-agnostic simulation:

* **`HUMAN_WORK_ITEM`**: Operations with positive processing duration ($> 0\text{s}$) that consume human worker capacity from discovered organizational roles and are subject to operational shift hours.
* **`AUTOMATED_TASK`**: Automated system operations with $0.0\text{s}$ duration that execute 24/7 without consuming human staff capacity, while retaining external/batch queue delays.
* **`INSTANT_GATEWAY`**: Pure routing/decision markers or status milestones with $0.0\text{s}$ duration and negligible wait time ($\le 60\text{s}$) that advance the Petri net marking instantaneously.

```bash
# Profile activity execution behaviors and generate classification manifest
python src/activity_profiler.py
```
* **Inputs**: `data/processed/aligned_<log_name>.xes` (or `<log_name>_filtered.xes`)
* **Outputs**: `models/activity_profiles.json` and `models/activity_profiles_report.md`

---

### Step 2: Log Alignment & Replay Repair (`src/02_align_and_repair_log.py`)

Replays the event log against the discovered Petri net using $A^*$ state-space alignment to produce a repaired log containing strictly synchronous moves. This guarantees that every trace in the dataset can be replayed inside the simulation model without deadlock:

```bash
python src/02_align_and_repair_log.py
```
* **Inputs**: `data/processed/<log_name>_filtered.xes` and `models/petri_nets/discovered_model_split.pnml`
* **Output**: `data/processed/aligned_<log_name>.xes`

---

### Step 2b: Causal Delay & Worker Latency Discovery (`src/02b_discover_delays.py`)

Isolates uncoupled external delays (e.g. customer response waiting, postal transit, external lab analysis) from resource contention and models human worker refractory latencies (post-task documentation, cognitive cooldowns, breaks):

* **Chapela-Campa & Dumas Estimators**: Evaluates candidate delays using `EclAw+`, `EclAw`, and `Naive` estimators (*Chapela-Campa & Dumas, 2022, 2024*).
* **Triple Causal Independence Tests**:
  1. *Congestion Independence*: Waiting duration is statistically uncorrelated with overall system WIP ($|r| < 0.30$).
  2. *Resource Availability*: Waiting occurs even when qualified resources are idle (un-eclipsed by worker busy intervals).
  3. *Transition Specificity*: Delay occurs consistently at specific control-flow transitions.
* **Continuous Kolmogorov–Smirnov Fitting**: Fits continuous parametric distributions (Lognormal, Gamma, Exponential) to statistically verified delay intervals.

```bash
python src/02b_discover_delays.py
```
* **Input**: `data/processed/aligned_<log_name>.xes`
* **Outputs**: `models/delays/delay_manifest.json` and `models/delays/delay_discovery_report.md`

---

### Step 3: Feature Engineering & Dataset Creation (`src/03_dataset_builder.py`)

Sweeps observed lifecycle events and estimated eligibility timestamps chronologically. Duration features are captured at START, after assignment; waiting features are captured at estimated eligibility, before the target starts. Step 02b supplies the eligibility estimate; this is not exact timed Petri-net replay. See [the feature conventions and limitations](docs/first16_fixes.md).

* **Universal Process Features**:
  * `Prefix`: Historical sequence of recent activities completed within the case (configurable window size $k$, default 10).
  * `Prev_Proc_Time`: Duration of the most recently completed activity in the case at the snapshot time.
  * `WIP`: Process-level Work-In-Progress (total active cases across the entire system at time $t$).
  * `AC_WIP`: Number of running instances of the target activity at time $t$; queued instances are excluded. The target is included at its duration snapshot and excluded at its waiting snapshot.
  * `Daytime`: Continuous normalized time-of-day feature in $[0, 1]$.
  * `Weekday_0` $\dots$ `Weekday_6`: One-hot indicators for day of the week.
* **Dynamic Departmental Occupancy (`Role_k_OC`)**:
  For the worker roster eligible for role $k$, divide occupied nominal capacity slots by total nominal slots:

$$
\text{Role}_k^{\text{OC}}(t) = \frac{\sum_{w \in R_k} \min(n_w(t), C_w)}{\sum_{w \in R_k} C_w}
$$

  Here $n_w(t)$ counts all running assignments of worker $w$, including assignments in other roles. $C_w$ is the worker's concurrency quota (one when quotas are disabled). This measures occupied work capacity, not calendar availability or unobserved breaks.
* **Declarative Case & Event Attributes**:
  Fits case-attribute imputation on training cases only. Waiting inputs may use case attributes known on arrival; eventual resource IDs and execution-event attributes are rejected for waiting prediction.
* **Cumulative Trace History (`trace_history`)**:
  Tracks snapshot-time completed-activity presence indicators (`has_<activity>`) for activities executed within the case prior to time $t$. Configured via `mode: "profile_aware"`, RIMS++ automatically monitors past occurrences of non-human activities (`INSTANT_GATEWAY` and `AUTOMATED_TASK`) discovered by the `ActivityProfiler`, providing duration/waiting predictors with historical process context. Routing has a separate extraction path; remaining routing inconsistencies are recorded in the audit.
* **Task-Specific Filtering**:
  * **Duration Dataset**: Filters out 0.0s events (automated transitions) and applies smoothed sample-weight corrections for filtering (`activity_weights.json`).
  * **Waiting Time Dataset**: Preserves 0.0s events (representing immediate resource pickups from queue).

```bash
# Generate duration dataset (0-duration filtered)
python src/03_dataset_builder.py --task duration

# Generate waiting time dataset (0-waiting preserved)
python src/03_dataset_builder.py --task waiting_time
```
* **Outputs**:
  * `data/processed/datasets/dataset_duration_global.csv` & `activity_weights.json`
  * `data/processed/datasets/dataset_waiting_time_global.csv`
  * Dataset `.summary.json` files record exclusions and partition counts; `.provenance.json` sidecars identify the source split and artifact contents.

---

### Duration clocks and external work

For paired executions, dataset creation calls `calculate_net_working_seconds` using the activity/role calendar. Simulation expands the predicted working duration with `calculate_calendar_duration`. This avoids the original blanket double-counting of nights and weekends, and was already present before the first-16 fixes.

There are remaining limitations:

- Duration extraction preserves raw elapsed time for same-day sessions up to four hours on working days (an overtime heuristic), and credits up to one hour if there is no working-calendar overlap. Runtime still applies shift pauses; these exceptions do not round-trip exactly.
- A positive START–COMPLETE interval is not proof of uninterrupted human work. Long asynchronous/external processing is currently classified as `HUMAN_WORK_ITEM` unless overridden, and consumes worker capacity.
- `AUTOMATED_TASK` means **zero execution duration plus possible waiting**, not nonzero continuous external processing. Overriding a long task to this type would discard its execution duration.
- A per-activity calendar override with `enabled: false` already supports continuous duration arithmetic for that activity, but does **not** remove its worker allocation. A positive-duration, worker-free external-task type is not implemented.

Therefore, the complete external-processing proposal in the old notes is not implemented. Changing its semantics requires deciding whether the logged interval represents staffed work, an external timer, or a combination. The remaining audit findings and `docs/first16_fixes.md` describe the repair boundary; the repository is not claimed to be free of all audit issues.

### Step 4: Predictive Modeling & Multi-Architecture Benchmark Suite

The predictive modeling engine (`src/04_train.py`) and benchmarking suite (`src/04b_benchmark_models.py`) support **3 Exploration Modes** across **both duration and waiting time**:

1. **`hybrid`** *(Recommended)*:
   Trains global and local candidates across architectures (`xgboost`, `tcn`, `lstm`) and automatically synthesizes a **Best-of-All-Worlds Hybrid Champion** dispatch configuration that routes each activity to its top-performing model.
2. **`global_tournament`**:
   Trains competing global models (Global XGBoost vs. Global TCN vs. Global LSTM) and selects the single best global architecture.
3. **`single_global`**:
   Fast single-model training with zero benchmark overhead.

The default temporal evaluation uses shared case-arrival cutoffs near 70% and 80%. Training cases must finish before the first cutoff; validation cases must arrive after it and finish before the second cutoff; test cases arrive after the second cutoff. Boundary-crossing cases are excluded, so the retained proportions are not exactly 70/10/20. Discovery and fitted preprocessing use training cases only. Champion selection uses validation MAE; the frozen champion is evaluated on test cases separately.

**Artifact rebuild required after the first-16 audit fixes:** old datasets and deployments have incompatible feature semantics. Run the pipeline from discovery onward; see [the migration and behavior notes](docs/first16_fixes.md).

#### Automated Benchmarking (`src/04b_benchmark_models.py`)

```bash
# 1. Full Hybrid Benchmark (Duration)
python src/04b_benchmark_models.py --task duration --mode hybrid --run_all

# 2. Full Hybrid Benchmark (Waiting Time)
python src/04b_benchmark_models.py --task waiting_time --mode hybrid --run_all

# 3. Global-Only Tournament
python src/04b_benchmark_models.py --task duration --mode global_tournament --run_all

# 4. Re-compare & synthesize champion from existing trained models
python src/04b_benchmark_models.py --task duration --compare
python src/04b_benchmark_models.py --task waiting_time --compare
```

#### Step-by-Step Individual Training (`src/04_train.py`)

Any individual model architecture can be trained independently:

```bash
# Duration Task
python src/04_train.py --task duration --strategy global --model_type xgboost
python src/04_train.py --task duration --strategy local --model_type xgboost
python src/04_train.py --task duration --strategy global --model_type tcn
python src/04_train.py --task duration --strategy local --model_type tcn
python src/04_train.py --task duration --strategy global --model_type lstm
python src/04_train.py --task duration --strategy local --model_type lstm

# Waiting Time Task
python src/04_train.py --task waiting_time --strategy global --model_type xgboost
python src/04_train.py --task waiting_time --strategy local --model_type xgboost
python src/04_train.py --task waiting_time --strategy global --model_type tcn
python src/04_train.py --task waiting_time --strategy local --model_type tcn
python src/04_train.py --task waiting_time --strategy global --model_type lstm
python src/04_train.py --task waiting_time --strategy local --model_type lstm
```

* **Outputs**:
  * Trained checkpoints and scalers: `models/{task}/{strategy}_{model_type}/`
  * Standalone deployment dispatch manifest: `models/{task}/hybrid_champion/dispatch_config.json`
  * Markdown benchmark report: `models/{task}/benchmark_report.md`

---

### Step 5: White-Box XOR Decision Mining & Routing (`src/05_train_routing.py`)

Discovers all XOR split places in the Petri net, extracts decision contexts from aligned event traces on `COMPLETE` events, and trains interpretable decision models with Hyperopt Bayesian optimization:

* **White-Box Interpretability**: Uses `DecisionTreeClassifier` with human-readable decision rules (`tree.export_text`) printed directly to stdout.
* **Data & History-Aware Features**: Classifiers leverage case payload attributes, inter-arrival time intervals, and dynamic trace history (`has_<activity>` presence flags of prior gateways and automated tasks) to capture business decision logic.
* **Hyperparameter Optimization**: Employs `hyperopt` with Tree-structured Parzen Estimator (TPE) across 100 iterations (`max_depth: 1..21`, `min_samples_split: 2..20`, `min_samples_leaf: 1..25`, `criterion`, `class_weight`).
* **Robust Empirical Fallback**: If $\text{Macro F1} < 0.60$ or an enabled transition has zero training samples, the router automatically falls back to empirical branching probabilities observed in the historical log.

```bash
# 1. Train Decision Trees for all XOR split places in the discovered Petri net
python src/05_train_routing.py --pnml models/petri_nets/discovered_model_split.pnml --log data/processed/aligned_<log_name>.xes

# Optional: Run with Random Forest ensemble (5 trees)
python src/05_train_routing.py --classifier random_forest --n_estimators 5

# Optional: Run with XGBoost classifier
python src/05_train_routing.py --classifier xgboost
```
* **Outputs**:
  * Routing dispatch manifest: `models/routing/routing_decisions.json`
  * Trained model binaries: `models/routing/routing_*.joblib`

---

## Step 6: Simulation Engine (`src/06_simulate.py` & `src/simulator/`)

The simulation engine is structured as a modular object-oriented package in `src/simulator/`, orchestrated via the CLI runner `src/06_simulate.py`:

* **`Token` (`src/simulator/token.py`)**: Case token advancing through Petri net places and transitions using formal PM4Py marking semantics ($M' = M - \text{Preset}(t) + \text{Postset}(t)$). Supports **true concurrent execution of parallel branches (AND-splits)** via SimPy subprocesses, evaluates XOR decision trees with empirical probability fallback for enabled transitions, applies loopback damping ($0.5^{\text{loop}}$), and provides safe deadlock detection warnings:
  * **3-Way Archetype Execution**: Coordinates with `ActivityProfiler` to execute transitions according to their operational archetype:
    * `INSTANT_GATEWAY`: Fires in $0.0\text{s}$ SimPy time without requesting workers or shift checks, advancing marking immediately.
    * `AUTOMATED_TASK`: Runs 24/7 without consuming human staff capacity, evaluating external/batch waiting time distributions while executing with $0.0\text{s}$ duration.
    * `HUMAN_WORK_ITEM`: Queues for human worker capacity from the mapped role, subject to operational shift calendars, multi-tasking limits, and empirical cooldown latencies.
  * **Causal Trace History Tracking**: Dynamically increments token-level history (`token.trace_history`) upon activity execution, ensuring decision routing and ML time predictors receive accurate, leak-free case state.
* **`SimulationProcess` & `SharedWorkerPool` (`src/simulator/process.py`)**: Coordinates the SimPy runtime environment and human resources using the **Hybrid Role-Resource Architecture**:
  * **Route by Role**: Tokens request resources from the activity's assigned organizational role (`activity_to_role`), querying qualified candidate workers.
  * **Zero Worker Ubiquity (Elimination of Phantom Staff)**: Tracks individual human workers as unique physical entities across the enterprise. If a multi-skilled worker is occupied with a task in Role A, they cannot be concurrently scheduled in Role B.
  * **Longest Idle First (LIF) Dispatching**: Selects the worker with the longest idle duration among free candidates, ensuring equitable work distribution.
  * **Multitasking Concurrency Quotas ($C_w$)**: Enforces empirical worker-level concurrency limits mined from the log, allowing natural task interleaving without creating artificial queue bottlenecks.
  * **Refractory Latency Cooldowns & Shift Alarms**: Models post-task rest breaks (Kolmogorov–Smirnov continuous distributions) and enforces business shift hours via an off-hours calendar gate and SimPy shift alarms (holding off-hours releases until 08:00 AM next business day).
* **`ChampionPredictor` (`src/simulator/predictor.py`)**: Unified runtime inference wrapper serving winning XGBoost, PyTorch TCN, and LSTM models from `dispatch_config.json`, dynamically aligning categorical data types via `categories.json` to prevent single-row encoding discrepancies.
* **`CalendarManager` (`src/simulator/calendar.py`)**: Manages operational shifts (e.g. 08:00 – 20:00 Mon–Fri) and custom per-role work schedules, pausing and advancing time over nights and weekends.
* **`ArrivalGenerator` (`src/simulator/arrivals.py`)**: Spawns cases via historical replay from the aligned log, synthetic generative distributions (exponential, uniform, constant), or non-linear time-series forecasting via **Facebook Prophet** (bounded logistic growth, diurnal/weekly seasonality, and sub-second model caching in `models/prophet/`). In Generative and Prophet modes, realistic case attributes are drawn via empirical bootstrap sampling from historical cases.
* **`SimulationEvaluator` (`src/simulator/metrics.py`)**: Exports flat CSV (`data/generated/csv/simulated_log.csv`) and standard IEEE XES (`data/generated/xes/simulated_log.xes`), and evaluates cycle time MAE, median, 1D Wasserstein distance, and Relative Event Distribution (RED / Earth Mover's Distance) against the defined historical cohort. Failed metrics are unavailable with diagnostics; completed-case metrics exclude censored/failed cases and are labeled conditional on completion.

```bash
# 1. Run 2-way comparison study (Pure Physics vs. Hybrid Residual side-by-side)
python src/06_simulate.py --compare_modes --cases 1000

# 2. Simulate with Prophet time-series arrivals (Hybrid Residual mode, 1,000 cases, cached model)
python src/06_simulate.py --cases 1000 --arrival_mode prophet

# Force retraining of Prophet arrival model ignoring disk cache
python src/06_simulate.py --cases 1000 --arrival_mode prophet --force_retrain_prophet

# 3. Simulate with Historical Replay arrivals
python src/06_simulate.py --cases 1000 --arrival_mode replay

# 4. Purely Generative arrivals (statistical distribution)
python src/06_simulate.py --cases 1000 --arrival_mode generative

# 5. Simulate ALL cases in historical log
python src/06_simulate.py --all

# 6. Pure Physics mode (SimPy queue contention only)
python src/06_simulate.py --cases 1000 --mode pure_physics

# 7. Set simulation horizon by duration (e.g. 365 days / 1 year)
python src/06_simulate.py --duration 365 --arrival_mode generative
```

* **Outputs**:
  * Simulated event log (CSV): `data/generated/csv/simulated_log.csv`
  * Simulated event log (IEEE XES): `data/generated/xes/simulated_log.xes`
  * Case statuses: `data/generated/csv/simulated_log_cases.csv`
  * Machine-readable results and metric statuses: `data/processed/simulation_metrics.json`
  * Markdown benchmark report: `data/processed/simulation_report.md`
  * Ablation study leaderboard: `data/processed/ablation_report.md`

---

## Validation Methodology & Multi-Perspective Evaluation Suite

RIMS++ directly integrates the official [`log-distance-measures`](https://github.com/AutomatedProcessImprovement/log-distance-measures) library (**Chapela-Campa et al., *BPM* 2023**), matching the benchmarking framework established by **Meneghello et al. (*Information Systems* 128, 2025)** and FBK's **WISDOM** pipeline across four foundational dimensions:

```
                                       RIMS++ Evaluation Suite
       ┌───────────────────────────┬──────────────┴──────────────┬───────────────────────────┐
       ▼                           ▼                             ▼                           ▼
  Control-Flow                  Temporal                     Workforce                  Congestion
  ├─ CFLD (Edit + Hungarian)    ├─ AED (Timeline 1WD)        └─ CWD (168h Staff Active) ├─ CTD (Cycle Time 1WD)
  ├─ NGD (2-Gram & 3-Gram)      ├─ CED (168h Circadian Week)                            ├─ CAR (Case Arrival 1WD)
  └─ L_traces (Length Diff)     └─ RED (Relative Trace Start)                           └─ Mean, Median, MAE
```

Replay references are restricted to scheduled case IDs. Full-case replay metrics use paired IDs completed on both sides within the observation window. Generated-arrival references use the corresponding historical arrival window. Completed-case scores are conditional on completion; separate window metrics use clipped observed intervals. Missing/error metrics are `null` with a status, not zero. A hard horizon does not drain unfinished cases.

### 1. Control-Flow Perspective
* **Control-flow Log Distance (CFLD)**: Measures structural trace divergence by computing the normalized string edit distance (Damerau-Levenshtein) between trace variants and solving the minimum-weight bipartite matching via the **Hungarian algorithm** (Kuhn 1955).
* **Trace Length Error ($L^{traces}$)**: Average absolute difference in case length between paired traces in $\mathcal{L}_{sim}$ and $\mathcal{L}_{real}$ under optimal Hungarian matching.
* **N-Gram Distribution Distance (NGD)**: Compares transition sequence probabilities for $N=2$ (2-Gram / Directly-Follows Graph) and $N=3$ (3-Gram) using Total Variation Distance / 1WD bounded in $[0.0, 1.0]$.

### 2. Temporal Perspective
* **Absolute Event Distribution (AED)**: Bins all event timestamps across the entire absolute calendar timeline into 1-hour intervals and computes the 1st Wasserstein Distance (1WD / Earth Mover's Distance in hours).
* **Circadian Event Distribution (CED)**: Projects event timestamps onto a 168-hour weekly grid ($7 \text{ days} \times 24 \text{ hours}$) to evaluate whether work shifts and diurnal activity rhythms match historical reality.
* **Relative Event Distribution (RED)**: Evaluates intra-case pacing by calculating 1WD between event offsets relative to case origin ($t_{event} - t_{case\_start}$).

### 3. Workforce / Resource Scheduling Perspective
* **Circadian Workforce Distribution (CWD)**: Measures the concurrency of active resources across the 168-hour weekly cycle ($7 \text{ days} \times 24 \text{ hours}$), computing 1WD between the real and simulated staff availability profiles.

### 4. Congestion & Cycle Time Perspective
* **Cycle Time Distribution (CTD / 1WD)**: Stored in seconds (`wd` is an alias); the library’s hourly-bin result is converted before reporting. Measures the minimum work required to transform the simulated cycle time distribution into the true historical cycle time distribution:
  $$W_1(P, Q) = \int_{-\infty}^{+\infty} |F_P(x) - F_Q(x)| \, dx$$
* **Case Arrival Rate (CAR)**: Evaluates arrival burstiness and concurrency by computing 1WD across binned case arrival timestamps.
* **Cycle Time Mean, Median & MAE**: Completed-cohort turnaround summaries. The current `mae_cycle` field is the absolute difference between the two cohort means, not a mean of per-case paired errors.

### Simulation comparison reports (`data/processed/ablation_report.md`)

`--compare_modes` reports measured scores for `hybrid_residual` and `pure_physics`, displaying unavailable scores as `N/A`. It does not assume either mode is better, promise a runtime, or award an overall best-fit star across different completed-case populations. Consult each run's population and censoring counts alongside its scores.
