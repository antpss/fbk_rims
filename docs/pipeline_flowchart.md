# RIMS+ End-to-End Pipeline Flowchart

This document provides a comprehensive flowchart and architectural breakdown of the **RIMS+** (Runtime Integration of Machine Learning & Simulation) end-to-end operational pipeline.

---

## 1. End-to-End Operational Flowchart

```mermaid
flowchart TD
    %% Styling Classes
    classDef data fill:#1f6feb,stroke:#58a6ff,color:#fff;
    classDef process fill:#238636,stroke:#2ea043,color:#fff;
    classDef model fill:#8957e5,stroke:#a371f7,color:#fff;
    classDef sim fill:#d29922,stroke:#bb8009,color:#fff;
    classDef eval fill:#da3633,stroke:#f85149,color:#fff;

    %% Step 0: Input & Filtering
    RawLog["Raw Event Log<br/>(BPI_Challenge_2012.xes)"]:::data
    Step0["Step 0: Log Filtering & Preprocessing<br/>(src/00_filter_log.py)"]:::process
    RawLog --> Step0
    
    FilteredLog["Filtered Work Items Log<br/>(BPI_2012_filtered.xes)"]:::data
    MilestoneLog["Milestone Lifecycle Log<br/>(BPI_2012_A_only.xes)"]:::data
    Step0 --> FilteredLog
    Step0 --> MilestoneLog

    %% Step 1: Discovery
    Step1["Step 1: Process Discovery<br/>(src/01_process_discovery.py)<br/>Split Miner / Inductive Miner"]:::process
    FilteredLog --> Step1
    DiscoveredPN["Workflow Petri Net<br/>(discovered_model_split.pnml)"]:::model
    Step1 --> DiscoveredPN

    %% Step 2: Alignment & Repair
    Step2["Step 2: Log Alignment & Repair<br/>(src/02_align_and_repair_log.py)<br/>State-Equation A* Replay"]:::process
    FilteredLog --> Step2
    DiscoveredPN --> Step2
    AlignedLog["Repaired & Aligned Log<br/>(aligned_BPI_2012.xes)<br/>Fitness = 100%"]:::data
    Step2 --> AlignedLog

    %% Step 2b: Delays
    Step2b["Step 2b: Causal Delays & Latencies<br/>(src/02b_discover_delays.py)<br/>Triple Causal Test + KS Fitting"]:::process
    AlignedLog --> Step2b
    DelayManifest["Causal Delay Manifest<br/>(models/delays/delay_manifest.json)"]:::model
    Step2b --> DelayManifest

    %% Step 3: Feature Engineering
    Step3["Step 3: Dataset Builder<br/>(src/03_dataset_builder.py)<br/>Dynamic State Feature Extraction"]:::process
    AlignedLog --> Step3
    
    DurationDS["Duration Dataset<br/>(dataset_duration_global.csv)<br/>Zero-durations filtered"]:::data
    WaitDS["Waiting Time Dataset<br/>(dataset_waiting_time_global.csv)<br/>Zero-wait preserved"]:::data
    Step3 --> DurationDS
    Step3 --> WaitDS

    %% Step 4: ML/DL Model Training
    Step4["Step 4: Predictive Modeling Engine<br/>(src/04_train.py & src/benchmark_models.py)<br/>TCN, XGBoost, LSTM (Global vs. Local)"]:::process
    DurationDS --> Step4
    WaitDS --> Step4
    ChampionSuite["Champion Dispatch Suites<br/>(models/duration/hybrid_champion/<br/>models/waiting_time/hybrid_champion/)"]:::model
    Step4 --> ChampionSuite

    %% Step 5: XOR Decision Mining
    Step5["Step 5: White-Box XOR Decision Mining<br/>(src/05_train_routing.py)<br/>Decision Trees / RF + Hyperopt TPE"]:::process
    AlignedLog --> Step5
    DiscoveredPN --> Step5
    RoutingModels["Routing Decision Manifest & Trees<br/>(models/routing/routing_decisions.json)"]:::model
    Step5 --> RoutingModels

    %% Step 6: Discrete-Event Simulation
    Step6["Step 6: SimPy Discrete-Event Simulation Engine<br/>(src/06_simulate.py & src/simulator/)"]:::sim
    
    DiscoveredPN --> Step6
    ChampionSuite --> Step6
    RoutingModels --> Step6
    DelayManifest --> Step6

    %% Simulation Internal Modules
    subgraph SimEngine["SimPy DES Runtime Architecture"]
        ArrivalGen["Arrival Generator<br/>- Replay historical<br/>- Generative (exp/unif)<br/>- Facebook Prophet"]:::sim
        WorkerPool["Shared Worker Pool<br/>- 59 multi-skilled agents<br/>- Calendar working shifts<br/>- Refractory cooldowns"]:::sim
        TokenProc["Token Marking Engine<br/>- PM4Py Petri net firing<br/>- XOR ML classification"]:::sim
        Predictor["Champion Predictor<br/>- Runtime TCN/XGBoost<br/>- Dynamic state features"]:::sim
        HybridWait["Hybrid Residual Waiting<br/>Queue contention + max(0, W_ML - W_queue)"]:::sim

        ArrivalGen --> TokenProc
        WorkerPool <--> TokenProc
        TokenProc --> Predictor
        TokenProc --> HybridWait
    end

    Step6 --- SimEngine

    %% Output & Evaluation
    SimCSV["Simulated Log (CSV)<br/>(simulated_log.csv)"]:::data
    SimXES["Standard IEEE XES Log<br/>(simulated_log.xes)"]:::data
    SimReport["Simulation Benchmark Report<br/>Cycle MAE, RMSE, Wasserstein"]:::eval

    Step6 --> SimCSV
    Step6 --> SimXES
    Step6 --> SimReport
```

---

## 2. Stage Breakdown & Module Responsibilities

### Step 0: Log Preprocessing & Filtering ([`src/00_filter_log.py`](file:///home/antonio/fbk_rims/src/00_filter_log.py))
* **Goal**: Prepares high-quality event logs from raw data.
* **Actions**:
  * Translates Dutch activity names into English.
  * Separates human work items (`W_`) into `data/processed/BPI_2012_filtered.xes`.
  * Isolates business application lifecycle events (`A_`, `O_`) into `data/processed/BPI_2012_A_only.xes`.

---

### Step 1: Process Discovery ([`src/01_process_discovery.py`](file:///home/antonio/fbk_rims/src/01_process_discovery.py))
* **Goal**: Extracts a mathematically sound workflow Petri net.
* **Algorithms**:
  * **Split Miner** (default): High precision, discovers concurrency and loops.
  * **Inductive Miner**: Fallback option guaranteeing block-structured soundness.
* **Output**: `models/petri_nets/discovered_model_split.pnml`.

---

### Step 2: Log Alignment & Repair ([`src/02_align_and_repair_log.py`](file:///home/antonio/fbk_rims/src/02_align_and_repair_log.py))
* **Goal**: Resolves divergences between the empirical event log and the discovered process net.
* **Mechanism**:
  * Replays traces using PM4Py state-equation $A^*$ alignments.
  * Filters non-synchronous log moves and reconstructs conformant lifecycle executions.
  * Ensures 100% replay fitness so that simulated cases never encounter deadlocks.
* **Output**: `data/processed/aligned_BPI_2012.xes`.

---

### Step 2b: Causal Delay & Latency Discovery ([`src/02b_discover_delays.py`](file:///home/antonio/fbk_rims/src/02b_discover_delays.py))
* **Goal**: Isolates external uncoupled delays (customer transit / postal delays) and worker cooldowns.
* **Mechanism**:
  * **Triple Causal Test**: Resource Footprint Test $\to$ WIP/Congestion Correlation Test $\to$ Calendar Invariance (24/7 Clock) Test.
  * Fits parametric Kolmogorov-Smirnov continuous distributions (Lognormal, Gamma, Exponential).
* **Output**: `models/delays/delay_manifest.json`.

---

### Step 3: Feature Engineering & Tabular Datasets ([`src/03_dataset_builder.py`](file:///home/antonio/fbk_rims/src/03_dataset_builder.py))
* **Goal**: Constructs state-vector training tabular datasets without future lookahead bias.
* **Features Extracted**:
  * **Prefix**: Sequence of recent activities (up to window length 10).
  * **WIP / AC_WIP**: Total active cases in the process and active instances performing the specific task.
  * **Role Occupancy (`Role_*_OC`)**: Utilization ratio per organizational pool.
  * **Calendar**: Normalized time of day (`Daytime` $0.0 \dots 1.0$) and one-hot `Weekday_0..6`.
  * **Payloads**: Domain case attributes (e.g. `RequestedAmount`).
* **Task Isolation**:
  * **Duration**: Filters out instantaneous 0.0s automated events; calculates smoothed sample weights.
  * **Waiting Time**: Preserves 0.0s events (representing immediate queue pickups).

---

### Step 4: Multi-Architecture Predictive Modeling ([`src/04_train.py`](file:///home/antonio/fbk_rims/src/04_train.py) & [`src/benchmark_models.py`](file:///home/antonio/fbk_rims/src/benchmark_models.py))
* **Goal**: Evaluates multiple machine learning / deep learning architectures on strict case-level splits (70/10/20 train/val/test).
* **Engines Evaluated**:
  * **TCN** (PyTorch Temporal Convolutional Network with causal dilated convolutions).
  * **XGBoost** (Gradient boosted trees with `log1p` target transform).
  * **LSTM** (PyTorch stacked recurrent baseline).
* **Strategies**:
  * Global (pooled across all activities) vs. Local (specialized per activity).
  * **Hybrid Champion**: Automatically selects the winning architecture per activity (TCN for sequence-heavy, XGBoost for tabular volume).
* **Output**: `models/{task}/hybrid_champion/dispatch_config.json`.

---

### Step 5: White-Box XOR Decision Mining ([`src/05_train_routing.py`](file:///home/antonio/fbk_rims/src/05_train_routing.py))
* **Goal**: Determines path choices at Petri net XOR decision points dynamically.
* **Mechanism**:
  * Extracts decision contexts from aligned logs at places with out-degree $\ge 2$.
  * Trains interpretable white-box models (`DecisionTreeClassifier` or `RandomForestClassifier`) optimized via Hyperopt Bayesian TPE (`max_depth: 1..21`).
  * **Fallback**: If Macro F1 $< 0.60$, automatically falls back to empirical branching probabilities.
* **Output**: `models/routing/routing_decisions.json` and model binaries (`.joblib` / `.pkl`).

---

### Step 6: Discrete-Event Simulation Runtime ([`src/06_simulate.py`](file:///home/antonio/fbk_rims/src/06_simulate.py) & [`src/simulator/`](file:///home/antonio/fbk_rims/src/simulator/))
* **Goal**: Full closed-loop discrete-event process simulation with live ML runtime queries.
* **Key Components**:
  * **Arrival Generation**: Historical replay, parametric generative distributions, or **Facebook Prophet** hourly time-series forecasting.
  * **Resource Contention**: 59 distinct human agents respecting business shift calendars (08:00–20:00 Mon–Fri).
  * **Token Semantics**: PM4Py formal marking traversal, resolving XOR decisions via trained trees.
  * **Champion Inference**: Unified predictor querying winning TCN, XGBoost, or LSTM models.
  * **Hybrid Residual Waiting**: Workers acquired in SimPy queue contention ($W_{\text{queue}}$); simulator delays only uncoupled residual $\max(0, W_{\text{ML}} - W_{\text{queue}})$.
* **Evaluation & Outputs**:
  * `simulated_log.csv` and standard IEEE `simulated_log.xes`.
  * Cycle time MAE, RMSE, and 1D Wasserstein distance benchmark report.

