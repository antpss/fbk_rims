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
├── models/                            # Discovered Petri nets and baseline models
│   ├── discovered_model_split.pnml    # Discovered Petri net (Split Miner)
│   ├── discovered_model_inductive.pnml# Discovered Petri net (Inductive Miner)
│   ├── tcns/                          # Trained PyTorch TCN models & scalers
│   ├── xgboost/                       # Trained XGBoost models
│   ├── lstms/                         # Trained PyTorch LSTM models & scalers
│   └── xgboost_routing.json           # Trained XOR routing classifier
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
* **Outputs**: `models/discovered_model_split.pnml`, `models/discovered_model_inductive.pnml`

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
# Generates zero-filtered datasets in data/processed/datasets_no_zeros/
python 03b_dataset_creation_no_zeros.py

# (Optional) Generates unfiltered datasets in data/processed/datasets/
python 03_dataset_creation.py
```
* **Outputs**: `data/processed/datasets_no_zeros/dataset_duration_[activity_name].csv`

### Step 5: Model Training

#### A. Duration Prediction Models
Activity-specific regression models predicting task duration in seconds. Use `--no_zeros` to train on the zero-filtered datasets:

```bash
# 1. Temporal Convolutional Networks (PyTorch TCN)
python 04c_train_tcns.py --no_zeros

# 2. Gradient Boosted Trees (XGBoost)
python 04d_train_xgboost.py --no_zeros

# 3. Long Short-Term Memory (PyTorch LSTM)
python 04b_train_lstms.py --no_zeros
```
* **Outputs**:
  * Model files: `tcn_[activity].pt`, `xgboost_[activity].json`, `lstm_[activity].pt`
  * Scalers: `scaler_[activity].pkl` (StandardScaler fitted on the 19 feature columns)
  * Token Vocabulary: `vocab.json` (activity string to integer embedding mapping)

#### B. Routing / Decision Point Models
Resolves XOR branching in the Petri net using machine learning classifiers conditioned on case attributes and execution prefix:

```bash
cd ../RIMS_decision_points
python decision_mining.py
```
* **Output**: `models/xgboost_routing.json`


