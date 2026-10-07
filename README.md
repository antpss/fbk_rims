# RIMS+ Pipeline: Runtime Integration of Machine Learning & Simulation

This repository contains the advanced **RIMS+** (Runtime Integration of Machine Learning and Simulation) framework for Business Process Simulation, specifically tailored for scenarios with high resource contention. It combines white-box Petri net process models executed via **SimPy** with machine learning and deep learning models (**TCN**, **XGBoost**, **LSTM**, and **Transformer**) predicting activity execution durations and queue waiting times at runtime, alongside decision classifiers resolving XOR branching.

---

### Repository Structure

```
fbk_rims/
├── config.yaml                        # Central configuration (tasks, features, paths, models)
├── requirements.txt                   # Formal project dependencies (PyTorch, SimPy, Hyperopt, PM4Py)
├── data/
│   ├── raw/                           # Raw input event logs (BPI_Challenge_2012.xes)
│   ├── processed/                     # Processed logs and tabular datasets
│   │   ├── BPI_2012_filtered.xes      # Filtered event log (work items only)
│   │   ├── BPI_2012_A_only.xes        # Filtered event log (application lifecycle for routing)
│   │   ├── aligned_BPI_2012.xes       # Repaired & aligned event log against Petri net
│   │   └── datasets/                  # Feature-engineered tabular datasets
│   │       ├── dataset_duration_global.csv     # Duration dataset (0-duration filtered)
│   │       ├── dataset_waiting_time_global.csv # Waiting time dataset (0-waiting preserved)
│   │       └── activity_weights.json           # Smoothed importance sample weights
│   └── generated/                     # Output logs from simulation runs
├── models/                            # Process models and predictive ML models
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   ├── discovered_model_split.pnml# Split Miner workflow net (W_ activities)
│   │   └── bpi2012_A_net.pnml         # Lifecycle Petri net for XOR decision mining (A_ activities)
│   ├── duration/                      # Activity processing duration models (XGBoost, TCN, LSTM)
│   │   ├── global_xgboost/            # Pooled global gradient boosted trees
│   │   ├── local_xgboost/             # Specialized local trees per activity
│   │   ├── global_tcn/                # Pooled global sequence TCN network
│   │   ├── local_tcn/                 # Specialized local sequence TCN networks
│   │   ├── global_lstm/               # Pooled global sequence LSTM network
│   │   ├── local_lstm/                # Specialized local sequence LSTM networks
│   │   ├── hybrid_champion/           # Dynamic Champion Hybrid dispatch & scalers
│   │   └── benchmark_report.md        # Duration benchmark leaderboard & activity breakdown
│   ├── waiting_time/                  # Activity waiting time / queue models (XGBoost, TCN, LSTM)
│   │   ├── hybrid_champion/           # Waiting time champion dispatch & scalers
│   │   └── benchmark_report.md        # Waiting time benchmark leaderboard
│   └── routing/                       # White-box XOR decision classifiers & routing manifest
│       ├── routing_decisions.json     # Dynamic decision dispatch manifest
│       └── p_*.pkl                    # Trained decision tree classifiers per XOR place
├── src/                               # Core pipeline scripts (Steps 00 to 06 + Benchmarking)
│   ├── 00_filter_log.py               # Step 0: Raw event log filtering
│   ├── 01_process_discovery.py        # Step 1: Petri net discovery (Split / Inductive)
│   ├── view_models.py                 # Step 1b: Petri net visualization (.png)
│   ├── 02_align_and_repair_log.py     # Step 2: Log alignment & replay repair
│   ├── 02b_conformance_checking.py    # Step 2b: Conformance evaluation (Fitness, Precision)
│   ├── 03_dataset_builder.py          # Step 3: Feature engineering & state vector extraction
│   ├── 04_train.py                    # Step 4: Multi-architecture training engine (XGB, TCN, LSTM)
│   ├── benchmark_models.py            # Step 4b: Automated benchmarking & champion synthesis
│   ├── 05_train_routing.py            # Step 5: White-Box XOR decision mining & Hyperopt tuning
│   ├── 06_simulate.py                 # Step 6: Discrete-event simulation engine
│   └── config_loader.py               # Central config loader helper
```

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
* `torch` : Neural network architectures (**LSTM**, **TCN**, and **Transformer**)
* `xgboost` : Gradient-boosted regression engines
* `scikit-learn` : DecisionTreeClassifier, feature scalers, and evaluation metrics
* `hyperopt` : Bayesian Tree-structured Parzen Estimator (TPE) optimization
* `pandas`, `numpy` : High-performance tabular manipulation and state extraction
* `pyyaml` : Centralized configuration parsing

---

## Central Configuration (`config.yaml`)

The entire pipeline is driven by [`config.yaml`](config.yaml):

* **`paths`**: Decoupled, non-conflicting paths for dual-process modeling:
  * Duration & Waiting Time (`W_` work items): `raw_log`, `aligned_log`, `petri_net`.
  * XOR Decision Routing (`A_` lifecycle): `routing_raw_log`, `routing_petri_net`, `routing_models_dir`.
* **`discovery`**: Miner algorithm (`split` or `inductive`) and noise parameters.
* **`preprocessing`**: Activity prefixes (`process_activity_prefixes`), milestone prefixes (`milestone_prefixes`), and Dutch-to-English translation mapping (`activity_mapping`).
* **`features`**: Context features to extract (WIP, activity WIP, role occupancy, calendar time, prefix window length, domain attributes like `RequestedAmount`).
* **`tasks`**: Task-specific data settings:
  * `duration`: Filters 0.0s events (automated pass-throughs) and calculates smoothed sample weights.
  * `waiting_time`: Preserves 0.0s events (immediate resource pickups).
  * `routing`: Decision mining settings (`classifier_type`, `max_depth: 21`, `min_samples: 30`, `min_f1_threshold: 0.60`).
* **`model_strategy`**:
  * `mode`: Training mode (`hybrid`, `global_tournament`, or `single_global`).
  * `single_global_engine`: Engine for single global training (`xgboost`, `tcn`, `lstm`).
  * `dl`: PyTorch neural network hyperparameters (`epochs: 50`, `patience: 5`, `batch_size: 256`, `learning_rate: 0.001`).
  * `candidates`: Candidate architectures evaluated for hybrid champion synthesis and global tournament.

---

## End-to-End Operational Pipeline

All commands should be executed from the project root directory.

### Step 0: Log Preprocessing & Filtering (`src/00_filter_log.py`)
Filters the raw BPI Challenge 2012 log to keep human work items (`W_` events) and standardizes lifecycle transitions.

```bash
python src/00_filter_log.py
```
* **Input**: `data/raw/BPI_Challenge_2012.xes`
* **Output**: `data/processed/BPI_2012_W_only.xes`

---

### Step 1: Process Discovery (`src/01_process_discovery.py`)
Discovers a structurally sound workflow Petri net from the filtered log and verifies soundness via Woflan.

```bash
# Split Miner (Recommended)
python src/01_process_discovery.py --miner split

# Alternatively, Inductive Miner
python src/01_process_discovery.py --miner inductive --noise_threshold 0.2
```
* **Outputs**: `models/petri_nets/discovered_model_split.pnml` (or `inductive.pnml`)

To generate visual diagram images of the discovered nets:
```bash
python src/view_models.py
```
* **Outputs**: `models/petri_nets/split_miner_visual.png`, `models/petri_nets/inductive_miner_visual.png`

---

### Step 2: Log Alignment & Repair (`src/02_align_and_repair_log.py`)
Replays the event log against the discovered Petri net using A* state-space alignment to produce a repaired log containing strictly synchronous moves, guaranteeing that every trace can be replayed inside the simulation model without deadlock.

```bash
python src/02_align_and_repair_log.py
```
* **Output**: `data/processed/aligned_BPI_2012.xes`

To evaluate conformance (Fitness, Precision, F1-Score):
```bash
python src/02b_conformance_checking.py --miner split
```

---

### Step 3: Feature Engineering & Dataset Creation (`src/03_dataset_builder.py`)
Sweeps the aligned log chronologically to build comprehensive state vectors without future data leakage:
* **Universal Context Features (Tier 1)**:
  * `Prefix`: Sequence of previous activities (history window up to 10 events).
  * `Prev_Proc_Time`: Duration of the immediate previous activity in the case.
  * `WIP`: Total active cases currently in the process (system-wide congestion).
  * `AC_WIP`: Active cases performing this specific activity (activity bottleneck).
  * `Daytime`: Normalized time of day ($0.0 \dots 1.0$).
  * `Weekday_0` $\dots$ `Weekday_6`: One-hot binary indicators for day of the week.
  * `Role_0_OC` $\dots$ `Role_5_OC`: Resource occupancy percentage per role pool.
* **Domain Case Attributes (Tier 2)**:
  * `RequestedAmount`: Loan amount requested (`case:AMOUNT_REQ`).
* **Task-Specific Filtering**:
  * **Duration**: Filters out 0.0s instant events (automated pass-throughs/missing timestamps) and applies smoothed inverse sample weights (`activity_weights.json`).
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
   Trains competing global engines (e.g. Global XGBoost vs. Global TCN vs. Global LSTM) and crowns the single best global model across the whole process.
3. **`single_global`**: 
   Fast single-model baseline (e.g. Global XGBoost, Global TCN, or Global LSTM) with zero benchmark overhead.

All models are evaluated on a strict **Case-Level 70/10/20 train/val/test split** (partitioned by `Case_ID` to eliminate cross-event data leakage).

#### Automated Benchmarking (`src/benchmark_models.py`)

> **Note**: `src/benchmark_models.py --run_all` is the automated orchestrator. It calls `src/04_train.py` under the hood for each candidate architecture in sequence, evaluates them on held-out test cases, and synthesizes the champion.

```bash
# 1. Full Hybrid Benchmark (Duration)
python src/benchmark_models.py --task duration --mode hybrid --run_all

# 2. Full Hybrid Benchmark (Waiting Time)
python src/benchmark_models.py --task waiting_time --mode hybrid --run_all

# 3. Fast Global-Only Tournament (XGBoost vs. TCN vs. LSTM)
python src/benchmark_models.py --task duration --mode global_tournament --run_all

# 4. Re-compare & synthesize champion from existing trained models
python src/benchmark_models.py --task duration --compare
```

#### Step-by-Step Training (`src/04_train.py`)

You can train any specific model architecture individually for either task:

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
Discovers all XOR split places (places with $\ge 2$ outgoing branches) in the lifecycle Petri net (`bpi2012_A_net.pnml`), aligns the event log using PM4Py state-equation $A^*$ alignments to extract exact decision contexts, and trains interpretable decision models with Hyperopt Bayesian optimization:

* **White-Box Interpretability**: Uses `DecisionTreeClassifier` with human-readable decision rules (`tree.export_text`) directly printed to stdout.
* **Hyperparameter Optimization (RIMS Parity)**: Employs `hyperopt` with Tree-structured Parzen Estimator (TPE) across 100 evaluations:
  * `max_depth`: $1 \dots 21$
  * `min_samples_split`: $2 \dots 20$
  * `min_samples_leaf`: $1 \dots 25$
  * `criterion`: `['gini', 'entropy', 'log_loss']`
  * `class_weight`: `[None, 'balanced']`
* **Fallback Strategy**:
  * If $\text{Macro F1} \ge 0.60$: Saves trained white-box model (`models/routing/{place}.pkl` and `routing_{place}.joblib`).
  * If $\text{Macro F1} < 0.60$ (or highly skewed classes): Automatically falls back to empirical branching probabilities from the aligned log.

```bash
# Standard run (reads config.yaml, optimizes DecisionTreeClassifier via Hyperopt)
python src/05_train_routing.py

# Optional: Run with Random Forest ensemble (5 trees)
python src/05_train_routing.py --classifier random_forest --n_estimators 5

# Optional: Run with XGBoost
python src/05_train_routing.py --classifier xgboost

# Override max depth ceiling or sample threshold
python src/05_train_routing.py --max_depth 21 --min_samples 30 --min_f1_threshold 0.60
```
* **Outputs**:
  * Modern dispatch manifest: `models/routing/routing_decisions.json`
  * Trained white-box model binaries: `models/routing/p_*.pkl` and `models/routing/routing_p_*.joblib`

---

## Offline Benchmark Results (Duration Task)

Evaluated on held-out test cases (1,932 unseen cases, 11,609 events):

### 1. Leaderboard

| Configuration | Strategy | Engine | Test MAE (s) | Test SMAPE (%) | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **hybrid_champion** | `hybrid` | `champion` | **647.25s** | **80.84%** | 1077.66s |
| local_tcn | `local` | `tcn` | 647.97s | 81.16% | 496.47s |
| local_xgboost | `local` | `xgboost` | 648.35s | 81.27% | 4.14s |
| global_tcn | `global` | `tcn` | 649.14s | 81.49% | 573.65s |
| global_xgboost | `global` | `xgboost` | 649.24s | 81.90% | 3.40s |

### 2. Hybrid Champion Architecture Assignment

The dynamic champion selects the winning foundation model per activity:

| Activity | Winning Pillar | Engine | Test MAE (s) | Best SMAPE (%) |
| :--- | :--- | :---: | :---: | :---: |
| **W_Assess fraud** | `local_tcn` | **TCN** | **1316.23s** | 58.91% |
| **W_Call after incomplete files** | `global_xgboost` | **XGBOOST** | **901.53s** | 98.20% |
| **W_Call after offers** | `local_tcn` | **TCN** | **533.38s** | 69.39% |
| **W_Complete application** | `local_xgboost` | **XGBOOST** | **458.96s** | 81.87% |
| **W_Handle leads** | `local_tcn` | **TCN** | **1132.69s** | 76.64% |
| **W_Validate application** | `local_tcn` | **TCN** | **866.76s** | 89.64% |
| *Fallback (`__default__`)* | `global_tcn` | **TCN** | — | — |

* **Deep Learning (TCN) dominates complex sequence activities**: Wins 4 out of 6 activities.
* **XGBoost dominates high-volume tabular patterns**: Wins 2 activities (`W_Complete application`, `W_Call after incomplete files`).

---

## Waiting Time Task (Ready for Execution)

The waiting time pipeline operates symmetrically to duration, but with domain-specific physics:
* **Preserves 0.0s waiting times**: An activity that starts immediately upon enablement experiences 0 waiting time. Dropping 0s would introduce severe upward bias.
* **Run Waiting Time 5-Pillar Benchmark**:
  ```bash
  python src/benchmark_models.py --task waiting_time --mode hybrid --run_all
  ```
* **Output**:
  * Standalone models in `models/waiting_time/`
  * Champion dispatch table in `models/waiting_time/hybrid_champion/dispatch_config.json`
  * Leaderboard in `models/waiting_time/benchmark_report.md`

---

## Step 6: Modern Discrete-Event Simulation (`src/06_simulate.py`)

The modernized discrete-event simulation engine in `src/06_simulate.py` combines the theoretical rigor of Petri net token marking semantics with modern machine learning:

1. **True Petri Net Token Marking Semantics**: Loads any discovered Petri Net (`.pnml`), starts at initial marking $M_0$, resolves enabled transitions dynamically using `pm4py.objects.petri_net.semantics`, and fires transitions until the sink marking is reached.
2. **Dual Arrival Modes (Replay & Generative)**:
   - **Replay Mode**: Replays historical case arrival timestamps and loan payloads directly from the aligned event log.
   - **Generative Mode**: Generates purely synthetic cases from scratch using statistical inter-arrival distributions (exponential, uniform, constant) bounded by office hour arrival calendars (matching original RIMS `InterTriggerTimer`).
3. **Multi-Skilled Shared Worker Pool**: Accurately tracks all 59 distinct human employees with dynamic role proficiencies, eliminating the phantom worker bug.
4. **Per-Role & Global Calendar Shift Engine**: Supports global office hours (08:00 – 17:00 Mon–Fri) as well as custom per-role work shift overrides, dynamically pausing and rolling active work over nights and weekends.
5. **Dynamic Champion Inference**: Dispatches winning models per activity from `models/duration/hybrid_champion/dispatch_config.json` and `models/waiting_time/hybrid_champion/dispatch_config.json`.
6. **Agnostic Hybrid Residual Waiting Time**: Reconciles physical queue contention with ML predictions: $\text{Extra Wait} = \max(0, W_{\text{ML}} - W_{\text{queue}})$.
7. **XOR Decision Routing**: Uses trained XGBoost/Decision Tree classifiers (`models/routing/routing_decisions.json`) to route tokens across branch choices.
8. **Standard IEEE XES Export**: Exports both flat CSV and standard IEEE XES event logs (`simulated_log.xes`), allowing direct import into ProM, Disco, Celonis, and PM4Py.
9. **Automatic Real vs. Simulated Benchmark Evaluation**: Computes Cycle Time MAE, Wasserstein distance, and activity execution breakdown, writing reports to `data/processed/simulation_report.md`.

```bash
# 1. 3-Way Ablation Study (compares Pure Physics vs. Pure ML vs. Hybrid Residual)
python src/06_simulate.py --compare_modes --cases 2500

# 2. Replay historical arrivals (Hybrid Residual mode, 1,000 cases)
python src/06_simulate.py --cases 1000 --arrival_mode replay

# 3. Purely Generative arrivals from scratch (e.g. 1,000 synthetic cases)
python src/06_simulate.py --cases 1000 --arrival_mode generative

# 4. Simulate ALL cases available in historical log
python src/06_simulate.py --all

# 5. Pure Physics mode (SimPy queue contention only)
python src/06_simulate.py --cases 1000 --mode pure_physics

# 6. Pure ML mode (ML predicted wait only)
python src/06_simulate.py --cases 1000 --mode pure_ml
```

* **Outputs**:
  * Simulated event log (CSV): `data/processed/simulated_log.csv`
  * Simulated event log (IEEE XES): `data/processed/simulated_log.xes`
  * Markdown benchmark report: `data/processed/simulation_report.md`
  * Ablation study leaderboard: `data/processed/ablation_report.md`

---

