# RIMS+ Pipeline

This repository contains the advanced RIMS+ pipeline (Runtime Integration of Machine Learning and Simulation), utilizing Transformer networks (todo) and PM4Py.

## Project Structure
- `src/` - Contains the Python pipeline scripts.
- `data/` - Contains the event logs, with subdirectories `raw/` for input logs and `processed/` for aligned logs and ML datasets.
- `models/` - Output directory for discovered Petri nets (.pnml) and trained machine learning models.
- `RIMS_tool/` - The cloned reference simulation engine.

## Pipeline Steps

### Step 1: Process Discovery
This step parses the raw event log, runs a discovery algorithm to extract the business logic, guarantees structural soundness, and exports an executable Petri net.

**Command to run:**
First, navigate to the `src` directory, then run the script.

```bash
cd src

# Run with the default Split Miner
python 01_process_discovery.py

# Alternatively, run with Inductive Miner
python 01_process_discovery.py --miner inductive
```

**Outputs:**
- `models/discovered_model_split.pnml`
- `models/discovered_model_inductive.pnml`

### Step 2: Conformance Checking
Aligns the log with the discovered Petri net to validate the model's structural soundness, calculating Fitness, Precision, and F1-Score (using fast token-based replay or full alignments).

**Command to run (Split Miner is default):**
```bash
python 02_conformance_checking.py
```

### Step 3: Log Alignment & Repair
Aligns the event log with the discovered Petri net model and repairs it to enforce strict compatibility (keeps synchronous moves, forces model moves, drops log moves).

**Command to run:**
```bash
python 02b_repair_log.py
```
**Outputs:**
- `data/processed/aligned_BPI_2012.xes`

### Step 4: Dataset Creation
Parses the event log to create localized datasets for training Machine Learning models (Transformers). It extracts trace prefixes and calculates processing durations for every unique activity.

**Command to run:**
```bash
python 03_dataset_creation.py
```
**Outputs:**
- `data/processed/dataset_duration_[activity_name].csv`

*More steps (Model Training, Simulation Execution, etc.) will be documented here as they are developed.*
