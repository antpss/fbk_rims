# RIMS+ Pipeline: Runtime Integration of Machine Learning & Simulation

This repository contains the advanced **RIMS+** (Runtime Integration of Machine Learning and Simulation) framework for Business Process Simulation. It combines white-box Petri net process models executed via **SimPy** with machine learning and deep learning models (**TCN**, **XGBoost**, **LSTM**, and **Transformer**) predicting activity durations and queue waiting times at runtime, alongside decision classifiers resolving XOR branching.

---

## Repository Structure

```
fbk_rims/
├── config.yaml                        # Central configuration (tasks, features, paths, models)
├── data/
│   ├── raw/                           # Raw input event logs (BPI_Challenge_2012.xes)
│   ├── processed/                     # Processed logs and tabular datasets
│   │   ├── BPI_2012_W_only.xes        # Filtered event log (work items only)
│   │   ├── aligned_BPI_2012.xes       # Repaired & aligned event log against Petri net
│   │   └── datasets/                  # Feature-engineered tabular datasets
│   │       ├── dataset_duration_global.csv     # Duration dataset (0-duration filtered)
│   │       ├── dataset_waiting_time_global.csv # Waiting time dataset (0-waiting preserved)
│   │       └── activity_weights.json           # Smoothed importance sample weights
│   └── generated/                     # Output logs from simulation runs
├── models/                            # Process models and predictive ML models
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   ├── discovered_model_split.pnml
│   │   ├── discovered_model_inductive.pnml
│   │   ├── split_miner_visual.png
│   │   └── inductive_miner_visual.png
│   ├── duration/                      # Activity processing duration models (5 Pillars)
│   │   ├── global_xgboost/            # Pillar 1: Pooled global gradient boosted trees
│   │   ├── local_xgboost/             # Pillar 2: Specialized local trees per activity
│   │   ├── global_tcn/                # Pillar 3: Pooled global sequence TCN network
│   │   ├── local_tcn/                 # Pillar 4: Specialized local sequence TCN networks
│   │   ├── hybrid_champion/           # Pillar 5: Dynamic Champion Hybrid dispatch & scalers
│   │   └── benchmark_report.md        # Duration benchmark leaderboard & activity breakdown
│   ├── waiting_time/                  # Activity waiting time / queue models (5 Pillars)
│   │   ├── hybrid_champion/           # Waiting time champion dispatch & scalers
│   │   └── benchmark_report.md        # Waiting time benchmark leaderboard
│   └── routing/                       # XOR split decision classifiers (.joblib, .json)
├── src/                               # Core pipeline scripts (Steps 00 to 05 + Benchmarking)
│   ├── 00_filter_log.py               # Step 0: Raw event log filtering
│   ├── 01_process_discovery.py        # Step 1: Petri net discovery (Split / Inductive)
│   ├── view_models.py                 # Step 1b: Petri net visualization (.png)
│   ├── 02_align_and_repair_log.py     # Step 2: Log alignment & replay repair
│   ├── 02b_conformance_checking.py    # Step 2b: Conformance evaluation (Fitness, Precision)
│   ├── 03_dataset_builder.py          # Step 3: Feature engineering & state vector extraction
│   ├── 04_train.py                    # Step 4: Multi-architecture training engine
│   ├── benchmark_models.py            # Step 4b: Automated 5-pillar benchmarking & champion synthesis
│   ├── 05_train_routing.py            # Step 5: XOR decision mining & routing classifiers
│   └── config_loader.py               # Central config loader helper
└── RIMS/                              # The SimPy-based discrete-event simulation engine
```

---

## Setup & Prerequisites

Activate the virtual environment:

```bash
source venv/bin/activate
```

Key dependencies:
* `pm4py` : Process discovery, conformance checking, and log alignment
* `simpy` : Discrete-event simulation engine
* `torch` : Temporal Convolutional Networks (TCN), LSTM, and Transformer models
* `xgboost` : Gradient-boosted duration regressors and routing classifiers
* `scikit-learn` : Preprocessing scalers, metrics, and decision trees
* `pandas`, `numpy` : Tabular data manipulation and feature engineering
* `pyyaml` : Centralized configuration parsing

---

## End-to-End Operational Pipeline

All pipeline commands are executed from the project root directory.

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

### Step 4: Predictive Modeling & 5-Pillar Benchmark Suite

The predictive modeling architecture supports **3 Exploration Modes** configured in `config.yaml` or via CLI:
1. **`hybrid`** *(Recommended)*: Trains the 4 foundation pillars and empirically synthesizes the **Best-of-All-Worlds Hybrid Champion** per activity.
2. **`global_tournament`**: Trains only competing global engines (e.g. Global XGBoost vs. Global TCN) and crowns the best global model across the whole process.
3. **`single_global`**: Fast single-model baseline (e.g. Global XGBoost in 3.4 seconds) with zero benchmark overhead.

All models are evaluated on a strict **Case-Level 70/10/20 train/val/test split** (partitioned by `Case_ID` to eliminate cross-event data leakage).

#### Automated Benchmarking (`src/benchmark_models.py`)

Run the full benchmark suite for duration or waiting time:

```bash
# 1. Full 5-Pillar Hybrid Benchmark (Duration)
python src/benchmark_models.py --task duration --mode hybrid --run_all

# 2. Full 5-Pillar Hybrid Benchmark (Waiting Time)
python src/benchmark_models.py --task waiting_time --mode hybrid --run_all

# 3. Fast Global-Only Tournament
python src/benchmark_models.py --task duration --mode global_tournament --run_all

# 4. Re-compare & synthesize champion from existing models
python src/benchmark_models.py --task duration --compare
```

#### Step-by-Step Training (`src/04_train.py`)

You can also train any specific model architecture individually:

```bash
# 1. Global XGBoost
python src/04_train.py --task duration --strategy global --model_type xgboost

# 2. Local XGBoost (models per activity)
python src/04_train.py --task duration --strategy local --model_type xgboost

# 3. Global TCN (sequence neural net)
python src/04_train.py --task duration --strategy global --model_type tcn

# 4. Local TCN (sequence neural nets per activity)
python src/04_train.py --task duration --strategy local --model_type tcn
```

* **Outputs**:
  * Models saved in `models/{task}/{strategy}_{model_type}/`
  * Standalone deployment dispatch: `models/{task}/hybrid_champion/dispatch_config.json`
  * Benchmark leaderboard report: `models/{task}/benchmark_report.md`

---

### Step 5: XOR Decision Mining & Routing (`src/05_train_routing.py`)
Discovers all decision point places (places with $\ge 2$ outgoing branches) in the Petri net, extracts decision contexts from the aligned log, and trains routing classifiers.

* If `Macro F1 >= min_f1_threshold` (default 0.60): Saves the machine learning classifier (`.joblib` / `.pkl`).
* If `Macro F1 < min_f1_threshold`: Falls back automatically to empirical branching probabilities.

```bash
# Standard run (reads config.yaml, trains XGBoost)
python src/05_train_routing.py

# Optional: Use an interpretable Decision Tree
python src/05_train_routing.py --classifier decision_tree

# Optional: Adjust minimum sample threshold
python src/05_train_routing.py --min_samples 50
```
* **Outputs**:
  * Modern dispatch manifest: `models/routing/routing_decisions.json`
  * Legacy RIMS compatibility: `models/routing/{project}_decision_points.json` and `{place_id}.pkl`

---

## Current Benchmark Highlights (Duration Task)

Evaluated on held-out test cases (11,609 events):

| Configuration | Strategy | Engine | Test MAE (s) | Test SMAPE (%) | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **hybrid_champion**  | `hybrid` | `champion` | **647.25s** | **80.84%** | 1077.66s |
| local_tcn | `local` | `tcn` | 647.97s | 81.16% | 496.47s |
| local_xgboost | `local` | `xgboost` | 648.35s | 81.27% | 4.14s |
| global_tcn | `global` | `tcn` | 649.14s | 81.49% | 573.65s |
| global_xgboost | `global` | `xgboost` | 649.24s | 81.90% | 3.40s |

In the champion configuration, **TCN wins 4 out of 6 activities** (`W_Assess fraud`, `W_Call after offers`, `W_Handle leads`, `W_Validate application`), while **XGBoost wins 2 activities** (`W_Complete application`, `W_Call after incomplete files`).

