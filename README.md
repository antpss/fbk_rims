# RIMS++: Runtime Integration of Machine Learning & Simulation

**RIMS++** is a Business Process Simulation (BPS) framework that combines Petri net process models with Machine Learning. It discovers process structure, organizational roles, external delays, and routing rules directly from event logs, and simulates process executions in SimPy.

What the framework does:
* **Discovers and Executes Petri Nets**: Mines sound workflow nets from the event log (via Split Miner) and executes them using token marking semantics in SimPy.
* **Predicts Durations & Waiting Times with ML**: Trains machine learning models (XGBoost, TCN, LSTM) to predict activity processing and waiting times based on real-time process state (WIP, time of day, queue congestion).
* **Mines Organizational Roles & Manages Workers**: Discovers roles from the log (*Burattin, Sperduti, & Veluscek, 2013*) and simulates named workers across departments, tracking individual availability, multi-tasking limits, and preventing workers from being in two places at once.
* **Identifies External Delays & Cooldowns**: Detects delays caused by external factors (e.g. waiting for a customer response) or worker post-task cooldowns using causal estimators (*Chapela-Campa & Dumas, 2022, 2024*).
* **Learns Decision-Based Routing**: Trains white-box decision trees at XOR split places to route cases based on data attributes and past trace history, falling back to probabilities when needed.
* **Models Case Arrivals**: Generates new cases through historical replay, statistical distributions, or time-series forecasting with Facebook Prophet.

---

## Repository Structure

```
fbk_rims/
├── config.yaml                        # Central configuration (paths, miner, roles, features, models, simulation)
├── requirements.txt                   # Formal project dependencies (PyTorch, SimPy, PM4Py, Prophet, XGBoost)
├── context.md                         # Detailed project context, evolution, and architectural blueprint
├── README.md                          # Primary reference documentation
├── docs/                              # Flowcharts and architecture documentation
│   └── causal_discovery_flowchart.md  # Triple causal test flowchart for extraneous delays
├── data/                              # Data directories
│   ├── raw/                           # Raw input event logs (.xes / .csv)
│   ├── processed/                     # Cleaned logs, training tables, and benchmark reports
│   │   ├── <log_name>_filtered.xes    # Preprocessed and standardized event log
│   │   ├── <log_name>_milestones.xes  # Optional lifecycle milestone log (for multi-stream logs)
│   │   ├── aligned_<log_name>.xes     # Repaired & aligned log (100% synchronous moves)
│   │   ├── datasets/                  # Feature-engineered tabular datasets
│   │   │   ├── dataset_duration_global.csv     # Duration dataset (0-duration filtered)
│   │   │   ├── dataset_waiting_time_global.csv # Waiting time dataset (0-waiting preserved)
│   │   │   └── activity_weights.json           # Smoothed inverse sample importance weights
│   │   ├── simulation_report.md       # Latest simulation run benchmark report
│   │   └── ablation_report.md         # Simulation ablation study leaderboard
│   └── generated/                     # Ephemeral simulation outputs (ignored by Git)
│       ├── csv/simulated_log.csv      # Output simulated event log (flat CSV)
│       └── xes/simulated_log.xes      # Output simulated event log (standard IEEE XES)
├── models/                            # Trained model artifacts and manifests
│   ├── role_mapping.json              # Discovered organizational role partition, rosters, and capacities
│   ├── role_discovery_report.md       # Organizational role discovery markdown report
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   └── discovered_model_split.pnml     # Workflow net discovered via Split Miner
│   ├── delays/                        # Causal delays and worker refractory latencies
│   │   ├── delay_manifest.json        # Machine-readable delay manifest with KS-fitted distributions
│   │   └── delay_discovery_report.md  # Statistical delay discovery report
│   ├── duration/                      # Activity duration models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Best-of-All-Worlds Hybrid Champion dispatch manifest
│   │   ├── global_{xgboost|tcn|lstm}/  # Global model checkpoints & scalers
│   │   ├── local_{xgboost|tcn|lstm}/   # Activity-specialized local model checkpoints & scalers
│   │   └── benchmark_report.md        # Duration benchmark leaderboard & activity breakdown
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
│   ├── 00_filter_log.py               # Step 0: Raw event log preprocessing, cleaning & prefix partitioning
│   ├── 01_process_discovery.py        # Step 1: Petri net discovery (Split Miner)
│   ├── 01b_discover_roles.py          # Step 1b: Organizational role mining
│   ├── view_models.py                 # Step 1c: Petri net visualization (.png export)
│   ├── 02_align_and_repair_log.py     # Step 2: Log alignment & replay repair (PM4Py A* state space)
│   ├── 02b_discover_delays.py         # Step 2b: Causal external delay & worker latency discovery
│   ├── 03_dataset_builder.py          # Step 3: Feature engineering & state vector extraction
│   ├── features.py                    # Declarative Feature Registry & context builder
│   ├── 04_train.py                    # Step 4: Multi-architecture training engine (XGB, TCN, LSTM)
│   ├── benchmark_models.py            # Step 4b: Automated benchmarking & champion synthesis
│   ├── 05_train_routing.py            # Step 5: White-Box XOR decision mining & Hyperopt tuning
│   ├── 06_simulate.py                 # Step 6: Discrete-event simulation CLI runner
│   ├── simulator/                     # Modular OOP Simulation Package
│   │   ├── __init__.py                # Package exports
│   │   ├── calendar.py                # CalendarManager (working shifts, night/weekend rollover)
│   │   ├── process.py                 # SimulationProcess & SharedWorkerPool
│   │   ├── token.py                   # Token (PM4Py Petri net marking semantics, XOR branching)
│   │   ├── predictor.py               # ChampionPredictor (Runtime XGBoost, TCN, LSTM inference)
│   │   ├── arrivals.py                # ArrivalGenerator (Replay, Generative, Prophet time-series)
│   │   └── metrics.py                 # SimulationEvaluator (IEEE XES/CSV export, Wasserstein, RED/EMD)
│   └── config_loader.py               # Central config loader helper
├── RIMS/                              # Cloned Reference / Legacy Repo (Meneghello et al. baseline, NOT the simulator)
└── RIMS_tool/                         # Cloned Reference / Legacy Repo (BPM 2023 demo tool, NOT the simulator)
```

> [!NOTE]
> **Reference / Baseline Repositories (`RIMS/` & `RIMS_tool/`)**:
> `RIMS/` and `RIMS_tool/` are cloned reference repositories of earlier research prototypes (Meneghello et al., using TensorFlow 1.x / Keras `.h5` LSTM models). **They are not used by the simulator**; they are retained strictly for historical reference and baseline comparison. The production RIMS++ pipeline lives entirely in `src/`.

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
    algorithm: "sperduti"           # Options: "sperduti", "manual"
    threshold: 0.35                 # Jaccard clustering threshold (0.35 -> macro departments; >= 0.60 -> 1 role per activity)
    output_file: "models/role_mapping.json"
    report_file: "models/role_discovery_report.md"
  simulation:
    mode: "hybrid_sperduti"         # Options: "hybrid_sperduti", "legacy_anonymous", "activity_isolated"
    dispatch_policy: "longest_idle_first"
    enable_concurrency_quotas: true # Enables empirical multi-tasking quotas (C_w >= 1) mined from log
```

### 4. Feature Engineering Registry
Controls the dynamic state features extracted chronologically from the event log:
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
    - case_attributes             # Domain-specific case payload attributes
    - milestones                  # Dynamic state flags (for multi-stream logs)
    # - resource_id               # Worker identity (optional)

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

### 5. Log Preprocessing & Multi-Stream Filtering
Handles activity standardization, lifecycle pairing, and optional stream partitioning:
```yaml
preprocessing:
  input_log: "data/raw/<raw_event_log>.xes"
  
  # Operational activities modeled for duration, waiting, and resource scheduling:
  # Leave empty [] to model ALL activities in standard single-lifecycle event logs.
  process_activity_prefixes: []

  # Milestone activities tracked strictly as contextual business state:
  # Leave empty [] if the log does not separate milestone states from operational tasks.
  milestone_prefixes: []

  # Optional activity renaming dictionary (e.g. translation or normalization):
  activity_mapping: {}
```

> [!NOTE]
> **Understanding Prefix Filtering (Standard Logs vs. Multi-Stream Logs)**:
> * **Standard Single-Lifecycle Logs (Default)**: In the vast majority of event logs (e.g., healthcare workflows, manufacturing tickets, IT incident logs), all recorded events represent operational tasks. In this setting, simply leave `process_activity_prefixes: []` and `milestone_prefixes: []`. Every activity in the log is modeled directly in the Petri net, trained for duration/waiting time, and assigned to organizational roles.
> * **Multi-Stream Event Logs (e.g., BPI Challenge 2012 / 2017)**: Certain enterprise logs record multiple interleaved event streams within the same trace — for instance, transactional human work items (e.g., prefixed with `W_`) alongside asynchronous case status milestones (e.g., application milestones prefixed with `A_` or offer updates prefixed with `O_`). For these logs, configuring `process_activity_prefixes: ["W_"]` and `milestone_prefixes: ["A_", "O_"]` cleanly decouples human work items for Petri net execution while transforming the asynchronous milestone streams into dynamic contextual features (`Milestone_<NAME>`), avoiding Petri net spaghetti models.

### 6. Config-Driven Task Specifications
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

### 7. Simulation Strategy & Environment
```yaml
simulation:
  mode: "hybrid_residual"         # Options: "hybrid_residual" (ML + physics) or "pure_physics"
  output_simulated_log: "data/generated/csv/simulated_log.csv"
  output_simulated_xes: "data/generated/xes/simulated_log.xes"
  extraneous_delays:
    enabled: true
    estimator: "eclaw_plus"       # Chapela-Campa & Dumas estimators: "eclaw_plus", "eclaw", "naive"
    placement: "ex_ante"
  human_latency:
    enabled: true
    max_cooldown: 1800.0          # Max post-task refractory pause / break (seconds)
  arrival:
    mode: "prophet"               # Options: "prophet", "replay", "generative"
    prophet:
      growth: "logistic"
      cap_multiplier: 1.25
      sampling_method: "triangular"
      diurnal_seasonality: true
      weekly_seasonality: true
      cache_dir: "models/prophet"
  calendar:
    business_hours:
      start: 8
      end: 20
    work_days: [0, 1, 2, 3, 4]    # Monday to Friday
```

---

## End-to-End Operational Pipeline

All pipeline steps can be executed sequentially from the project root with the virtual environment active:

```bash
source venv/bin/activate
```

---

### Step 0: Log Preprocessing & Filtering (`src/00_filter_log.py`)

Preprocesses raw event logs, standardizes activity names via configurable mappings, handles lifecycle transitions (`start` / `complete` event pairing to calculate processing duration and waiting times), and optionally partitions multi-stream logs:

```bash
python src/00_filter_log.py
```
* **Input**: `data/raw/<raw_event_log>.xes`
* **Outputs**:
  * `data/processed/<log_name>_filtered.xes` (clean operational event log)
  * `data/processed/<log_name>_milestones.xes` (optional milestone log when milestone prefixes are defined)

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
   $$S(a_i, a_j) = \frac{\sum_{r \in R} \min\big(p_i(r), p_j(r)\big)}{\sum_{r \in R} \max\big(p_i(r), p_j(r)\big)}$$
   where $p_i(r)$ denotes the number of times worker $r$ executed activity $a_i$.

2. **Agglomerative Department Partitioning**:
   * Controlled by the similarity threshold $\theta$ (`config.yaml: resources.discovery.threshold`):
     * A moderate threshold (e.g. $\theta = 0.35$) groups activities with shared personnel into macro functional departments.
     * A strict threshold (e.g. $\theta \ge 0.60$) separates tasks into highly specialized roles (approaching 1 role per activity).
   * Maps every activity to a discovered role, lists all qualified human workers, and calculates effective staffing capacity.

3. **Empirical Multi-Tasking Quotas ($C_w$)**:
   Computes the 95th percentile of concurrent active cases per worker directly from historical event intervals:
   $$C_w = \text{Percentile}_{95}\big(\text{active\_cases}(w, t)\big)$$
   This enables workers to interleave tasks naturally (e.g. desk office work, phone follow-ups) without causing artificial queue serialization.

```bash
# Discover organizational roles using configured similarity threshold θ
python src/01b_discover_roles.py --threshold 0.35
```
* **Inputs**: `data/processed/aligned_<log_name>.xes` (or `<log_name>_filtered.xes`)
* **Outputs**: `models/role_mapping.json` and `models/role_discovery_report.md`

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

Sweeps the aligned log chronologically to construct state feature vectors without future data leakage:

* **Universal Process Features**:
  * `Prefix`: Historical sequence of recent activities executed within the case (configurable window size $k$, default 10).
  * `Prev_Proc_Time`: Duration of the immediately preceding activity within the case.
  * `WIP`: Process-level Work-In-Progress (total active cases across the entire system at time $t$).
  * `AC_WIP`: Activity-Centric Work-In-Progress (cases active or queued for this specific activity at time $t$).
  * `Daytime`: Continuous normalized time-of-day feature in $[0, 1]$.
  * `Weekday_0` $\dots$ `Weekday_6`: One-hot indicators for day of the week.
* **Dynamic Departmental Occupancy (`Role_k_OC`)**:
  Computes the real-time utilization vector for each discovered organizational role:
  $$\text{Role\_k\_OC}(t) = \min\left(1.0, \; \frac{\text{Active Workers in Role } k \text{ at time } t}{\text{Effective Staffing Capacity of Role } k}\right)$$
  providing ML regressors with continuous macro-level departmental congestion signals.
* **Declarative Case & Event Attributes**:
  Extracts case-level attributes (e.g. case category, cost, requested amount, customer tier) with empirical median/mode imputation governed by `treat_zero_as_missing`.
* **Dynamic Milestone Tracking**:
  For multi-stream logs, tracks historical milestone indicators (`Milestone_<NAME>`) recording whether asynchronous lifecycle events have completed prior to time $t$.
* **Task-Specific Filtering**:
  * **Duration Dataset**: Filters out 0.0s events (automated transitions) and applies smoothed inverse sample weights (`activity_weights.json`).
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

---

### Step 4: Predictive Modeling & Multi-Architecture Benchmark Suite

The predictive modeling engine (`src/04_train.py`) and benchmarking suite (`src/benchmark_models.py`) support **3 Exploration Modes** across **both duration and waiting time**:

1. **`hybrid`** *(Recommended)*:
   Trains global and local candidates across architectures (`xgboost`, `tcn`, `lstm`) and automatically synthesizes a **Best-of-All-Worlds Hybrid Champion** dispatch configuration that routes each activity to its top-performing model.
2. **`global_tournament`**:
   Trains competing global models (Global XGBoost vs. Global TCN vs. Global LSTM) and selects the single best global architecture.
3. **`single_global`**:
   Fast single-model training with zero benchmark overhead.

All models are evaluated on a strict **Case-Level 70/10/20 train/val/test split** (partitioned by `Case_ID` to eliminate temporal data leakage).

#### Automated Benchmarking (`src/benchmark_models.py`)

```bash
# 1. Full Hybrid Benchmark (Duration)
python src/benchmark_models.py --task duration --mode hybrid --run_all

# 2. Full Hybrid Benchmark (Waiting Time)
python src/benchmark_models.py --task waiting_time --mode hybrid --run_all

# 3. Global-Only Tournament
python src/benchmark_models.py --task duration --mode global_tournament --run_all

# 4. Re-compare & synthesize champion from existing trained models
python src/benchmark_models.py --task duration --compare
python src/benchmark_models.py --task waiting_time --compare
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

## Step 6: Discrete-Event Simulation Engine (`src/06_simulate.py` & `src/simulator/`)

The simulation engine is structured as a modular object-oriented package in `src/simulator/`, orchestrated via the CLI runner `src/06_simulate.py`:

* **`Token` (`src/simulator/token.py`)**: Individual case token advancing through Petri net places and transitions using formal PM4Py marking semantics ($M' = M - \text{Preset}(t) + \text{Postset}(t)$), evaluating XOR decision trees with empirical probability fallback for enabled transitions, and applying loopback damping ($0.5^{\text{loop}}$).
* **`SimulationProcess` & `SharedWorkerPool` (`src/simulator/process.py`)**: Coordinates the SimPy runtime environment and human resources using the **Hybrid Role-Resource Architecture**:
  * **Route by Role**: Tokens request resources from the activity's assigned organizational role (`activity_to_role`), querying qualified candidate workers.
  * **Zero Worker Ubiquity (Elimination of Phantom Staff)**: Tracks individual human workers as unique physical entities across the enterprise. If a multi-skilled worker is occupied with a task in Role A, they cannot be concurrently scheduled in Role B.
  * **Longest Idle First (LIF) Dispatching**: Selects the worker with the longest idle duration among free candidates, ensuring equitable work distribution.
  * **Multitasking Concurrency Quotas ($C_w$)**: Enforces empirical worker-level concurrency limits mined from the log, allowing natural task interleaving without creating artificial queue bottlenecks.
  * **Refractory Latency Cooldowns & Shift Alarms**: Models post-task rest breaks (Kolmogorov–Smirnov continuous distributions) and enforces business shift hours via an off-hours calendar gate and SimPy shift alarms (holding off-hours releases until 08:00 AM next business day).
* **`ChampionPredictor` (`src/simulator/predictor.py`)**: Unified runtime inference wrapper serving winning XGBoost, PyTorch TCN, and LSTM models from `dispatch_config.json`, dynamically aligning categorical data types via `categories.json` to prevent single-row encoding discrepancies.
* **`CalendarManager` (`src/simulator/calendar.py`)**: Manages operational shifts (e.g. 08:00 – 20:00 Mon–Fri) and custom per-role work schedules, pausing and advancing time over nights and weekends.
* **`ArrivalGenerator` (`src/simulator/arrivals.py`)**: Spawns cases via historical replay from the aligned log, synthetic generative distributions (exponential, uniform, constant), or non-linear time-series forecasting via **Facebook Prophet** (bounded logistic growth, diurnal/weekly seasonality, and sub-second model caching in `models/prophet/`). In Generative and Prophet modes, realistic case attributes are drawn via empirical bootstrap sampling from historical cases.
* **`SimulationEvaluator` (`src/simulator/metrics.py`)**: Exports flat CSV (`data/generated/csv/simulated_log.csv`) and standard IEEE XES (`data/generated/xes/simulated_log.xes`), and evaluates cycle time MAE, median, 1D Wasserstein distance, and Relative Event Distribution (RED / Earth Mover's Distance) against historical ground truth.

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
  * Markdown benchmark report: `data/processed/simulation_report.md`
  * Ablation study leaderboard: `data/processed/ablation_report.md`

---

## Validation Methodology & Evaluation Metrics

RIMS++ evaluates simulation fidelity across both **macro process cycle time metrics** and **micro timestamp event distribution metrics**:

### 1. Macroscopic Cycle Time Metrics
* **Cycle Time MAE**: Mean Absolute Error between the simulated case durations and the ground-truth historical case durations.
* **Mean & Median Cycle Time**: Validates whether the simulation accurately replicates overall business turnaround times without systemic under- or over-estimation.
* **1D Wasserstein Distance (Earth Mover's Distance on Cycle Time)**: Measures the minimum work required to transform the simulated cycle time distribution into the true historical cycle time distribution:
  $$W_1(P, Q) = \int_{-\infty}^{+\infty} |F_P(x) - F_Q(x)| \, dx$$
  providing a robust, scale-sensitive distance metric between duration curves.

### 2. Microscopic Event Distribution Metrics
* **Relative Event Distribution Distance (RED / EMD)**: Evaluates the fine-grained temporal alignment of individual activity occurrences across the timeline. It partitions simulated and historical logs into discrete temporal bins (e.g. hourly or daily intervals) per activity and calculates the Earth Mover's Distance between their normalized event counts. A lower RED score indicates that activities occur at the realistic hours, days, and frequencies matching historical behavior.

### Illustrative Simulation Leaderboard Structure (`data/processed/ablation_report.md`)

When executing `--compare_modes`, RIMS++ generates a comprehensive comparison against the ground-truth historical benchmark:

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **REAL HISTORICAL** | *Ground Truth* | *Ground Truth* | — | — | — | — |
| **`hybrid_residual` (ML + Physics)** | **Close to Truth** | **Close to Truth** | **Lowest MAE** | **Lowest Wasserstein** | **Lowest RED** | ~150–200s |
| `pure_physics` (Contention Only) | Underestimated | Underestimated | Higher MAE | Higher Wasserstein | Higher RED | ~180–220s |


