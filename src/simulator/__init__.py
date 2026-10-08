#!/usr/bin/env python3
"""
src/simulator/__init__.py

Modular Discrete-Event Simulation Package for RIMS+.
Provides OOP components for:
  - CalendarManager (working shifts, night and weekend pauses)
  - SharedWorkerPool / WorkerManager (multi-skilled human resource engine)
  - ChampionPredictor (XGBoost, TCN, LSTM, Decision Tree routing)
  - Token (Petri net token marking semantics & life-cycle)
  - ArrivalGenerator (replay & generative arrival scheduling)
  - SimulationProcess / DiscreteEventSimulation (orchestrator engine)
  - SimulationEvaluator (metrics, XES/CSV export, reports)
"""

from .calendar import CalendarManager
from .predictor import ChampionPredictor
from .arrivals import ArrivalGenerator
from .token import Token
from .metrics import SimulationEvaluator, format_dur
from .process import SharedWorkerPool, WorkerManager, SimulationProcess, DiscreteEventSimulation

__all__ = [
    "CalendarManager",
    "ChampionPredictor",
    "ArrivalGenerator",
    "Token",
    "SimulationEvaluator",
    "format_dur",
    "SharedWorkerPool",
    "WorkerManager",
    "SimulationProcess",
    "DiscreteEventSimulation",
]
