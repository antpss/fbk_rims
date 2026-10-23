# RIMS+ Pipeline: Runtime Integration of Machine Learning & Simulation

This repository contains the advanced **RIMS+** (Runtime Integration of Machine Learning and Simulation) framework for Business Process Simulation (BPS), specifically tailored for operational scenarios characterized by high resource contention, non-linear worker behaviors, and complex uncoupled delays. It combines white-box Petri net process models executed via **SimPy** with runtime Machine Learning and Deep Learning inference engines (**XGBoost**, **TCN**, and **LSTM**), white-box XOR routing classifiers, continuous causal delay discovery, and Facebook Prophet time-series arrival forecasting.

---

### Repository Structure

```
fbk_rims/
├── config.yaml                        # Central configuration (paths, tasks, features, models, simulation)
├── requirements.txt                   # Formal project dependencies (PyTorch, SimPy, PM4Py, Prophet, XGBoost)
├── context.md                         # Detailed project context, evolution, and architectural blueprint
├── README.md                          # Comprehensive pipeline documentation and benchmark leaderboards
├── docs/                              # Flowcharts and architecture documentation
│   └── causal_discovery_flowchart.md  # Triple causal test flowchart for extraneous delays
├── data/                              # Data directories
│   ├── raw/                           # Raw event logs (e.g. BPI_Challenge_2012.xes)
│   ├── processed/                     # Cleaned logs, training tables, and benchmark reports
│   │   ├── BPI_2012_filtered.xes      # Filtered event log (human work items + milestones)
│   │   ├── BPI_2012_A_only.xes        # Application lifecycle milestone log
│   │   ├── aligned_BPI_2012.xes       # Repaired & aligned log (100% synchronous moves)
│   │   ├── datasets/                  # Feature-engineered tabular datasets
│   │   │   ├── dataset_duration_global.csv     # Duration dataset (0-duration filtered)
│   │   │   ├── dataset_waiting_time_global.csv # Waiting time dataset (0-waiting preserved)
│   │   │   └── activity_weights.json           # Smoothed sample importance weights
│   │   ├── simulation_report.md       # Latest simulation benchmark report
│   │   └── ablation_report.md         # Simulation ablation study leaderboard
│   └── generated/                     # Ephemeral simulation outputs (ignored by Git)
│       ├── csv/simulated_log.csv      # Output simulated event log (flat CSV)
│       └── xes/simulated_log.xes      # Output simulated event log (standard IEEE XES)
├── models/                            # Trained model artifacts and manifests
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   ├── discovered_model_split.pnml# Operational Split Miner workflow net (W_ activities)
│   │   └── bpi2012_A_net.pnml         # Milestone lifecycle net (A_ activities reference)
│   ├── delays/                        # Causal delays and worker refractory latencies
│   │   ├── delay_manifest.json        # Machine-readable delay manifest with KS-fitted distributions
│   │   └── delay_discovery_report.md  # Detailed delay discovery report
│   ├── duration/                      # Activity duration models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Synthesized Best-of-All-Worlds Hybrid Champion dispatch
│   │   ├── global_{xgboost|tcn|lstm}/  # Pooled global model checkpoints & scalers
│   │   ├── local_{xgboost|tcn|lstm}/   # Activity-specialized local model checkpoints & scalers
│   │   └── benchmark_report.md        # Duration benchmark leaderboard & activity breakdown
│   ├── waiting_time/                  # Activity waiting time models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Waiting time champion dispatch & scalers
│   │   ├── global_{xgboost|tcn|lstm}/  # Pooled global model checkpoints & scalers
│   │   ├── local_{xgboost|tcn|lstm}/   # Activity-specialized local model checkpoints & scalers
│   │   └── benchmark_report.md        # Waiting time benchmark leaderboard
│   ├── routing/                       # White-box XOR decision classifiers & routing manifest
│   │   ├── routing_decisions.json     # Dynamic decision dispatch manifest
│   │   └── routing_*.joblib           # Trained decision tree / random forest classifiers per XOR place
│   └── prophet/                       # Serialized Facebook Prophet arrival models & metadata
│       ├── prophet_arrivals_*.json    # Serialized model checkpoint for sub-second startup
│       └── prophet_arrivals_*_meta.json # Training metadata, log timestamp, and carrying capacity
├── src/                               # Active RIMS+ Core Pipeline & Simulator Package
│   ├── 00_filter_log.py               # Step 0: Raw event log filtering and activity renaming
│   ├── 01_process_discovery.py        # Step 1: Petri net discovery (Split / Inductive)
│   ├── view_models.py                 # Step 1b: Petri net visualization (.png)
│   ├── 02_align_and_repair_log.py     # Step 2: Log alignment & replay repair (PM4Py A*)
│   ├── 02b_discover_delays.py         # Step 2b: Causal external delays & worker latency discovery
│   ├── 03_dataset_builder.py          # Step 3: Feature engineering & state vector extraction
│   ├── features.py                    # Declarative Feature Registry & context builder
│   ├── 04_train.py                    # Step 4: Multi-architecture training engine (XGB, TCN, LSTM)
│   ├── benchmark_models.py            # Step 4b: Automated benchmarking & champion synthesis
│   ├── 05_train_routing.py            # Step 5: White-Box XOR decision mining & Hyperopt tuning
│   ├── 06_simulate.py                 # Step 6: Discrete-event simulation CLI runner
│   ├── simulator/                     # Modular OOP Simulation Package
│   │   ├── __init__.py                # Package exports
│   │   ├── calendar.py                # CalendarManager (working shifts, night/weekend rollover)
│   │   ├── process.py                 # SimulationProcess & SharedWorkerPool (59 human agents)
│   │   ├── token.py                   # Token (PM4Py Petri net marking semantics, XOR branching)
│   │   ├── predictor.py               # ChampionPredictor (Runtime XGBoost, TCN, LSTM inference)
│   │   ├── arrivals.py                # ArrivalGenerator (Replay, Generative, Prophet time-series)
│   │   └── metrics.py                 # SimulationEvaluator (IEEE XES/CSV export, Wasserstein MAE)
│   └── config_loader.py               # Central config loader helper
├── RIMS/                              # Cloned Reference / Legacy Repo (Meneghello et al. Keras baseline, NOT the simulator)
└── RIMS_tool/                         # Cloned Reference / Legacy Repo (BPM 2023 demo tool, NOT the simulator)
```

> [!NOTE]
> **Cloned Reference / Legacy Repositories (`RIMS/` & `RIMS_tool/`)**:
> `RIMS/` and `RIMS_tool/` are cloned reference repositories of previous research prototypes (Meneghello et al., using legacy TensorFlow 1.x / Keras `.h5` LSTM models on the `confidential_1000` dataset, and the BPM 2023 demo tool). **They are not the simulator**; they are kept strictly as reference codebases and comparison baselines. The active, production-grade RIMS+ simulator and operational pipeline live entirely in `src/`.

---

## Setup & Prerequisites

Activate the virtual environment and install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies:
* `pm4py` : Process discovery, conformance checking, and state-equation alignments
* `simpy` : Discrete-event simulation engine
* `torch` : Neural network architectures (**TCN** and **LSTM**)
* `xgboost` : Gradient-boosted regression and classification engines
* `scikit-learn` : DecisionTreeClassifier, RandomForestClassifier, feature scalers, and evaluation metrics
* `hyperopt` : Bayesian Tree-structured Parzen Estimator (TPE) optimization
* `prophet` : Time-series forecasting for case arrival intensities (DeepSimulator parity)
* `pandas`, `numpy`, `scipy` : Tabular manipulation, state extraction, and Kolmogorov-Smirnov distribution fitting
* `pyyaml` : Centralized configuration parsing

---

## Central Configuration (`config.yaml`)

The entire operational pipeline is governed by [`config.yaml`](config.yaml):

* **`paths`**: Input and output paths for event logs, discovered models, datasets, and manifests.
* **`discovery`**: Process miner selection (`split` or `inductive`) and noise thresholds.
* **`features`**: Context features to extract (`wip`, `ac_wip`, `daytime`, `weekday`, `role_occupancy`, `prev_duration`, `case_attributes`, `milestones`, `resource_id`), prefix window size (10), and declarative case attributes with `treat_zero_as_missing` and empirical median/mode imputation.
* **`preprocessing`**: Domain-agnostic activity prefixes (`process_activity_prefixes`), milestone prefixes (`milestone_prefixes`), and optional activity renaming mapping (`activity_mapping`). Supports empty `[]` to model all activities with zero prefix filtering.
* **`tasks`**: Task-specific data settings:
  * `duration`: Filters 0.0s events (automated pass-throughs) and calculates smoothed inverse sample weights (`activity_weights.json`).
  * `waiting_time`: Preserves 0.0s events (physically valid immediate resource pickups).
  * `routing`: Decision mining settings (`classifier_type: "decision_tree"`, `max_depth: 21`, `min_samples: 30`, `min_f1_threshold: 0.60`, `mode: "payload"`).
* **`model_strategy`**:
  * `mode`: Training mode (`hybrid`, `global_tournament`, or `single_global`).
  * `single_global_engine`: Engine for single global training (`xgboost`, `tcn`, `lstm`).
  * `dl`: PyTorch neural network hyperparameters (`epochs: 50`, `patience: 5`, `batch_size: 256`, `learning_rate: 0.001`, `min_delta: 0.01`).
  * `candidates`: Candidate architectures evaluated for hybrid champion synthesis and global tournaments.
* **`simulation`**:
  * `mode`: Waiting time strategy (`hybrid_residual` or `pure_physics`).
  * `output_simulated_log`: Dedicated flat CSV output path (`data/generated/csv/simulated_log.csv`).
  * `output_simulated_xes`: Dedicated standard IEEE XES output path (`data/generated/xes/simulated_log.xes`).
  * `extraneous_delays`: Chapela-Campa causal delay injection (`estimator: "eclaw_plus"`, `placement: "ex_ante"`).
  * `human_latency`: Queue-aware refractory cooldowns / coffee breaks (`max_cooldown: 1800.0s`).
  * `arrival`: Case spawning strategy (`replay`, `generative`, or `prophet`) with empirical bootstrap payload sampling for realistic synthetic arrivals.
    * `prophet`: Facebook Prophet time-series parameters (`growth: "logistic"` with dynamic `cap_multiplier: 1.25`, `sampling_method: "triangular"` or `"poisson"`, diurnal/weekly seasonalities, uniform/normal dispersion, and automatic model caching in `models/prophet/`).
  * `calendar`: Business operating hours (08:00 – 20:00 Mon–Fri) with strict off-hours dispatch gating (holding off-hours releases until 08:00 AM next business day) and custom `per_role` schedules.

---

## End-to-End Operational Pipeline


All commands should be executed from the project root with the virtual environment active:

```bash
source venv/bin/activate
```

### Step 0: Log Preprocessing & Filtering (`src/00_filter_log.py`)
Filters the raw event log, translates historical Dutch activity names to standard English, and exports dual logs:
* Operational human work items (`W_` events) with durations and resources.
* Application lifecycle milestones (`A_` and `O_` events) for clean control-flow discovery and dynamic business state tracking.

```bash
python src/00_filter_log.py
```
* **Input**: `data/raw/BPI_Challenge_2012.xes`
* **Outputs**: `data/processed/BPI_2012_filtered.xes` and `data/processed/BPI_2012_A_only.xes`

---

### Step 1: Process Discovery (`src/01_process_discovery.py`)
Discovers a structurally sound workflow Petri net from the filtered log and verifies soundness via Woflan:

```bash
# Split Miner (Recommended for high precision)
python src/01_process_discovery.py --miner split

# Alternatively, Inductive Miner
python src/01_process_discovery.py --miner inductive --noise_threshold 0.2
```
* **Output**: `models/petri_nets/discovered_model_split.pnml` (or `inductive.pnml`)

To render visual diagram images of all discovered nets:
```bash
python src/view_models.py
```
* **Outputs**: `models/petri_nets/*_visual.png`

---

### Step 2: Log Alignment & Repair (`src/02_align_and_repair_log.py`)
Replays the event log against the discovered Petri net using A* state-space alignment to produce a repaired log containing strictly synchronous moves, guaranteeing that every trace can be replayed inside the simulation model without deadlock:

```bash
python src/02_align_and_repair_log.py
```
* **Output**: `data/processed/aligned_BPI_2012.xes`

---

### Step 2b: Causal Delay & Worker Latency Discovery (`src/02b_discover_delays.py`)
Discovers uncoupled external delays (postal/client response transit) and human refractory inter-ticket latencies using Chapela-Campa & Dumas estimators (EclAw+, EclAw, Naive) and RIMS+ triple causal tests, fitting continuous Kolmogorov-Smirnov distributions (Lognormal, Gamma, Exponential):

```bash
python src/02b_discover_delays.py
```
* **Input**: `data/processed/aligned_BPI_2012.xes`
* **Outputs**: `models/delays/delay_manifest.json` and `models/delays/delay_discovery_report.md`

---

### Step 3: Feature Engineering & Dataset Creation (`src/03_dataset_builder.py`)
Sweeps the aligned log chronologically to construct state vectors without future data leakage:
* **Context Features**: `Prefix` (window size 10), `Prev_Proc_Time`, `WIP`, `AC_WIP`, `Daytime`, `Weekday_0` $\dots$ `Weekday_6`, `Role_0_OC` $\dots$ `Role_5_OC`, dynamic milestone indicators (`Milestone_<NAME>`), and domain-agnostic case attributes.
* **Empirical Imputation (`treat_zero_as_missing`)**: Precomputes empirical medians (for continuous attributes) and modes (for categorical attributes). When `treat_zero_as_missing: true`, $0.0$ placeholders (e.g., unrecorded loan amounts) are imputed with the median, while genuine structural zeroes are preserved when `false`.
* **Task-Specific Filtering**:
  * **Duration**: Filters out 0.0s events (automated pass-throughs/missing timestamps) and applies smoothed inverse sample weights (`activity_weights.json`).
  * **Waiting Time**: Preserves 0.0s events (representing immediate resource pickups from queue).

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
   Trains global and local candidates across architectures (`xgboost`, `tcn`, `lstm`) and empirically synthesizes the **Best-of-All-Worlds Hybrid Champion** per activity.
2. **`global_tournament`**:
   Trains competing global engines (Global XGBoost vs. Global TCN vs. Global LSTM) and crowns the single best global model.
3. **`single_global`**:
   Fast single-model baseline with zero benchmark overhead.

All models are evaluated on a strict **Case-Level 70/10/20 train/val/test split** (partitioned by `Case_ID` to eliminate cross-event data leakage).

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

#### Step-by-Step Training (`src/04_train.py`)

You can train any specific model architecture individually:

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
  * Models saved in task-isolated directories: `models/{task}/{strategy}_{model_type}/`
  * Standalone deployment dispatch manifest: `models/{task}/hybrid_champion/dispatch_config.json`
  * Markdown benchmark report: `models/{task}/benchmark_report.md`

---

### Step 5: White-Box XOR Decision Mining & Routing (`src/05_train_routing.py`)
Discovers all XOR split places in the Petri net, extracts decision contexts from aligned event traces on `COMPLETE` events, and trains interpretable decision models with Hyperopt Bayesian optimization:

* **White-Box Interpretability**: Uses `DecisionTreeClassifier` with human-readable decision rules (`tree.export_text`) printed to stdout.
* **Hyperparameter Optimization**: Employs `hyperopt` with Tree-structured Parzen Estimator (TPE) across 100 evaluations (`max_depth: 1..21`, `min_samples_split: 2..20`, `min_samples_leaf: 1..25`, `criterion`, `class_weight`).
* **Domain-Agnostic Context**: Dynamically tracks milestone states (`has_<ms>`), gates domain-specific offer features strictly behind `"O_"` prefix presence, and handles arbitrary continuous/categorical case attributes.
* **Empirical Fallback**: If $\text{Macro F1} < 0.60$ or an enabled transition is not matched, automatically routes using empirical branching probabilities derived from training traces.

```bash
# 1. Operational Workflow Net (default)
python src/05_train_routing.py --pnml models/petri_nets/discovered_model_split.pnml --log data/processed/aligned_BPI_2012.xes

# Optional: Run with Random Forest ensemble (5 trees)
python src/05_train_routing.py --classifier random_forest --n_estimators 5

# Optional: Run with XGBoost
python src/05_train_routing.py --classifier xgboost
```
* **Outputs**:
  * Modern dispatch manifest: `models/routing/routing_decisions.json`
  * Trained white-box model binaries: `models/routing/routing_*.joblib`

---

## Step 6: Discrete-Event Simulation Engine (`src/06_simulate.py` & `src/simulator/`)

The simulation engine is structured into the package `src/simulator/`, executed via `src/06_simulate.py`:

* **`Token` (`src/simulator/token.py`)**: Individual case token advancing through Petri net places and transitions using formal PM4Py marking semantics ($M' = M - \text{Preset}(t) + \text{Postset}(t)$), querying XOR decision trees with empirical probability fallback for enabled transitions, and applying loopback damping ($0.5^{\text{loop}}$).
* **`SimulationProcess` & `SharedWorkerPool` (`src/simulator/process.py`)**: Manages the SimPy runtime environment and the 59 distinct enterprise human resources. Dispatches via Longest Idle First (LIF), models continuous Kolmogorov-Smirnov refractory cooldowns (e.g. coffee breaks), and strictly enforces business shift hours via an off-hours calendar gate and SimPy shift alarms (holding off-hours releases until 08:00 AM next business day).
* **`ChampionPredictor` (`src/simulator/predictor.py`)**: Unified runtime inference wrapper serving winning XGBoost, PyTorch TCN, and LSTM models from `dispatch_config.json`, dynamically aligning categorical data types via `categories.json` to prevent single-row encoding discrepancies.
* **`CalendarManager` (`src/simulator/calendar.py`)**: Manages operational shifts (08:00 – 20:00 Mon–Fri) and custom per-role work schedules, pausing and advancing time over nights and weekends.
* **`ArrivalGenerator` (`src/simulator/arrivals.py`)**: Spawns cases via historical replay from `aligned_BPI_2012.xes`, synthetic generative distributions (exponential, uniform, constant), or time-series forecasting via **Facebook Prophet** (bounded logistic growth, triangular/poisson uncertainty sampling, and sub-second model caching in `models/prophet/`). In Generative and Prophet modes, realistic case attributes are drawn via empirical bootstrap sampling from historical cases.
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

### Simulation Validation Leaderboard (`data/processed/ablation_report.md`)

Evaluated against the ground-truth historical log (1,000 cases, Prophet arrival intensity):

| Simulation Mode | Mean Cycle Time | Median Cycle Time | Cycle Time MAE | Wasserstein Dist | RED Dist (EMD) | Run Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **REAL HISTORICAL** | **11.3 days** | **8.3 days** | — | — | — | — |
| **`hybrid_residual` ★ (Best)** | **10.2 days** | **4.9 days** | **1.2 days** | **3.6 days** | **191.47h** | 169.1s |
| `pure_physics` | 9.5 days | 3.1 days | 1.9 days | 4.7 days | 266.15h | 204.3s |

* **Realism & Congestion**: `hybrid_residual` captures unobserved worklist backlog, external postal/customer turnaround latency, and physical worker contention, achieving superior Wasserstein distance (3.6 days) and cycle time alignment (MAE: 1.2 days).
