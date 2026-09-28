# RIMS+ Pipeline: Runtime Integration of Machine Learning & Simulation

This repository contains the advanced **RIMS+** (Runtime Integration of Machine Learning and Simulation) framework for Business Process Simulation, specifically tailored for scenarios with high resource contention. It combines white-box Petri net process models executed via **SimPy** with deep learning and gradient boosted models (**TCN**, **XGBoost**, and **LSTM**) predicting activity processing durations at runtime, alongside decision trees resolving XOR branching.

---

## Repository Structure

```
├── data/
│   ├── raw/                           # Raw input event logs (BPI_Challenge_2012.xes)
│   ├── processed/                     # Aligned event logs and feature-engineered datasets
│   │   ├── aligned_BPI_2012.xes       # Repaired & aligned event log
│   │   ├── datasets/                  # Tabular datasets per activity (including 0-duration)
│   │   └── datasets_no_zeros/         # Tabular datasets per activity (0-duration filtered)
│   └── generated/                     # Output logs from simulation runs
├── models/                            # Process models and predictive ML models
│   ├── petri_nets/                    # Discovered Petri nets (.pnml) and visual diagrams (.png)
│   │   ├── discovered_model_split.pnml
│   │   ├── discovered_model_inductive.pnml
│   │   ├── split_miner_visual.png
│   │   └── inductive_miner_visual.png
│   ├── global_xgboost/                # Pillar 1: Global XGBoost model
│   ├── local_xgboost/                 # Pillar 2: Local XGBoost models per activity
│   ├── global_tcn/                    # Pillar 3: Global TCN sequence model
│   ├── local_tcn/                     # Pillar 4: Local TCN sequence models per activity
│   ├── hybrid_champion/              # Pillar 5: Dynamic Champion Hybrid dispatch
│   └── benchmark_report.md            # Offline model evaluation benchmark report
├── models_no_zeros/                   # Models trained on the zero-duration filtered datasets
│   ├── tcns/                          # TCN models, scalers, and vocab.json
│   ├── xgboost/                       # XGBoost models
│   └── lstms/                         # LSTM models, scalers, and vocab.json
├── src/                               # Core pipeline scripts (Discovery, Conformance, Datasets, Training)
├── RIMS/                              # The SimPy-based RIMS / RIMS+ discrete-event simulation engine
├── RIMS_decision_points/              # Decision mining and XOR branching scripts
└── context.md                         # Detailed project context and technical roadmap
```

---

## Setup & Prerequisites

Activate the virtual environment:

```bash
source venv/bin/activate
```

Key dependencies:
* `pm4py` — Process discovery, conformance checking, and log alignment
* `simpy` — Discrete-event simulation engine
* `torch` — Temporal Convolutional Networks (TCN) & LSTM models
* `xgboost` — Gradient-boosted duration regressors and routing classifiers
* `scikit-learn` — Preprocessing scalers, metrics, and decision trees
* `pandas`, `numpy` — Tabular data manipulation

---

## End-to-End Operational Pipeline

All pipeline scripts in `src/` should be executed with `src/` as the working directory:

```bash
cd src
```

### Step 1: Process Discovery
Parses the filtered event log, extracts the process structure, and exports a structurally sound Petri net.

```bash
# Split Miner (Recommended for precision)
python 01_process_discovery.py

# Alternatively, run with Inductive Miner
python 01_process_discovery.py --miner inductive
```
* **Outputs**: `models/petri_nets/discovered_model_split.pnml`, `models/petri_nets/discovered_model_inductive.pnml`

### Step 2: Conformance Checking
Evaluates the discovered Petri net against the event log to measure Fitness, Precision, and F1-Score.

```bash
python 02_conformance_checking.py
```

### Step 3: Log Alignment & Repair
Replays the event log against the Petri net to produce a repaired log containing strictly synchronous moves, ensuring full compatibility between the log and the simulation model.

```bash
python 02b_repair_log.py
```
* **Output**: `data/processed/aligned_BPI_2012.xes`

### Step 4: Dataset Creation
Processes the aligned log chronologically to engineer localized training datasets for each unique activity.
* **19 Dynamic Context Features**:
  * `Prefix`: Sequence of preceding activities (history up to length 10).
  * `RequestedAmount`: Monetary loan amount from case payload (`case:AMOUNT_REQ`).
  * `Prev_Proc_Time`: Duration of the immediate previous step in the case.
  * `WIP`: Total active cases in the system (global congestion).
  * `AC_WIP`: Active cases performing this specific activity (local bottleneck).
  * `Daytime`: Normalized time of day ($0.0 \dots 1.0$).
  * `Weekday_0` $\dots$ `Weekday_6`: One-hot binary indicators for day of the week.
  * `Role_0_OC` $\dots$ `Role_6_OC`: Resource occupancy percentage per role pool.
* **Instantaneous Event Filtering**: Filters out 0-duration human work items (`W_`) that represent automated pass-throughs or missing timestamps.

```bash
# Build duration dataset (filters 0.0s events, computes smoothed sample weights)
python src/03_dataset_builder.py --task duration

# Build waiting time dataset (retains 0.0s events for immediate pickups)
python src/03_dataset_builder.py --task waiting_time
```
* **Outputs**:
  * `data/processed/datasets_no_zeros/dataset_duration_global.csv` & `activity_weights.json`
  * `data/processed/datasets_no_zeros/dataset_waiting_time_global.csv`

---

### Step 4: Predictive Modeling Engine (`src/04_train.py`)

Unified multi-architecture training engine supporting **Global**, **Local**, and **Heterogeneous Multi-Tier Hybrid** strategies across XGBoost, TCN, LSTM, and Transformer.

```bash
# Train duration prediction (defaults to Heterogeneous Hybrid strategy from config.yaml)
python src/04_train.py --task duration

# Train waiting time prediction
python src/04_train.py --task waiting_time

# CLI Overrides:
python src/04_train.py --task duration --strategy global --model_type tcn
python src/04_train.py --task duration --strategy local --model_type xgboost
```
* **Outputs**:
  * Model files in `models_no_zeros/[strategy]/`
  * Runtime dispatch table: `models_no_zeros/[strategy]/dispatch_config.json`

---

### Step 5: XOR Decision Mining & Routing Classifier (`src/05_train_routing.py`)

Discovers all XOR branching places in the discovered Petri net, extracts decision contexts (sliding prefix window + case attributes), and trains classifiers with automatic empirical probability fallback.

```bash
# Standard run (reads config.yaml, trains XGBoost, evaluates Macro F1 against quality threshold)
python src/05_train_routing.py

# Optional CLI Overrides:
# 1. Use an interpretable Decision Tree instead of XGBoost:
python src/05_train_routing.py --classifier decision_tree

# 2. Adjust minimum sample threshold required before training an ML model:
python src/05_train_routing.py --min_samples 50
```
* **Outputs**:
  * Modern dispatch manifest: `models_no_zeros/routing/routing_decisions.json`
  * Legacy RIMS compatibility: `models_no_zeros/routing/[project]_decision_points.json` and `{place_id}.pkl`



