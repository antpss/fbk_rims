# Causal Delay Discovery Engine


---

## The Triple Causal Test Flowchart

```mermaid
flowchart TD
    Gap["Historical Delay between<br/>Activity A and Activity B"] --> T1["Test 1: Resource Footprint Test<br/>Is an employee assigned<br/>and logging hours?"]
    
    T1 -- "YES" --> Work["Physical Active Work<br/>(Handled by SimPy worker desks)"]
    
    T1 -- "NO" --> T2["Test 2: WIP / Congestion Test<br/>Does this delay stretch when<br/>the department gets busy?<br/>r(delay, WIP)"]
    
    T2 -- "YES (r > 0.40)" --> Queue["Internal Queue Contention<br/>(Handled by SimPy FIFO queues)"]
    
    T2 -- "NO (r ≈ 0)" --> T3["Test 3: Calendar Invariance Test<br/>Does this delay keep ticking<br/>across nights and weekends?<br/>(Off-hours ratio)"]
    
    T3 -- "NO (Pauses at 17:00)" --> Batch["Internal Business Hold / Batching"]
    
    T3 -- "YES (>= 50% off-hours)" --> EUD["External Uncoupled Delay (EUD)<br/>(Customer Postal Mail /<br/>Hospital Lab Incubation)"]
    
    EUD --> Fit["Step 4: Parametric Fitting via KS-Test<br/>(Lognormal, Gamma, Exponential)"]
    Fit --> Manifest["Saved to<br/>models/delays/delay_manifest.json"]

    style Gap fill:#1f6feb,stroke:#58a6ff,color:#fff
    style Work fill:#238636,stroke:#2ea043,color:#fff
    style Queue fill:#d29922,stroke:#bb8009,color:#fff
    style Batch fill:#8957e5,stroke:#a371f7,color:#fff
    style EUD fill:#da3633,stroke:#f85149,color:#fff
    style Fit fill:#0d419d,stroke:#388bfd,color:#fff
    style Manifest fill:#161b22,stroke:#8b949e,color:#fff
```

---

## Detailed Explanation of the 3 Causal Tests

### 1. Test 1: Resource Footprint Test
- **Question:** *Is an employee actively working during this interval?*
- **Mechanism:** Inspects whether `org:resource` is populated and active.
- **Verdict:** If an employee is logging hours, this time belongs to active work duration, not idle waiting.

### 2. Test 2: WIP / Congestion Test (Orthogonality)
- **Question:** *Does this delay stretch when the department gets busy?*
- **Mechanism:** Reconstructs continuous concurrent Work-in-Progress (WIP) and calculates the Pearson correlation $r(\text{delay}, \text{WIP})$.
- **Verdict:**
  - **Internal Queues:** When a department is flooded with tickets, queue times grow significantly ($r > 0.40$).
  - **External Delays:** A postal carrier or customer reading an offer at home takes 8.9 days regardless of whether the bank's internal desk queue is full or empty ($|r| < 0.25$).

### 3. Test 3: Calendar Invariance Test (The 24/7 Clock Test)
- **Question:** *Does this delay keep ticking during nights and weekends?*
- **Mechanism:** Samples points across the interval and calculates the percentage of time that elapsed during non-working hours (nights 17:00–08:00 and weekends).
- **Verdict:**
  - **Internal Office Tasks:** Pause at 17:00 and pause over weekends (off-hours $< 35\%$).
  - **External Delays:** Continue moving 24/7 over Saturday and Sunday (off-hours $\ge 50\%$).
