# RIMS+ Pipeline

This repository contains the advanced RIMS+ pipeline (Runtime Integration of Machine Learning and Simulation), utilizing Transformer networks (todo) and PM4Py.

## Project Structure
- `src/` - Contains the Python pipeline scripts.
- `models/` - Output directory for discovered Petri nets (.pnml).
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

*More steps (Conformance Checking, Transformer Training, etc.) will be documented here as they are developed.*
