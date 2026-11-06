"""
src/features.py

Declarative Feature Registry and Context Manager for RIMS++.
Enables clean, YAML-driven selection of feature sets across dataset generation,
model training, and simulation runtime inference.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable


@dataclass
class FeatureContext:
    """
    Unified context capturing process, case, and resource state at any execution step.
    Shared identically across offline dataset building and online simulation inference.
    """
    target_act: str = ""
    prefix: List[str] = field(default_factory=list)
    wip: float = 0.0
    ac_wip: Dict[str, int] = field(default_factory=dict)
    dt: Optional[datetime] = None
    requested_amount: float = 0.0
    role_oc: List[float] = field(default_factory=list)
    prev_dur: float = 0.0
    worker_id: str = "UNKNOWN"
    trace_history: Dict[str, int] = field(default_factory=dict)
    tracked_activities: List[str] = field(default_factory=list)
    case_attrs: Dict[str, Any] = field(default_factory=dict)
    event_attrs: Dict[str, Any] = field(default_factory=dict)
    include_counts: bool = False


def _extract_wip(ctx: FeatureContext) -> Dict[str, float]:
    return {"WIP": float(ctx.wip)}


def _extract_ac_wip(ctx: FeatureContext) -> Dict[str, float]:
    return {"AC_WIP": float(ctx.ac_wip.get(ctx.target_act, 0))}


def _extract_daytime(ctx: FeatureContext) -> Dict[str, float]:
    if ctx.dt is None:
        return {"Daytime": 0.0}
    daytime = (ctx.dt.hour * 3600 + ctx.dt.minute * 60 + ctx.dt.second) / 86400.0
    return {"Daytime": float(daytime)}


def _extract_weekday(ctx: FeatureContext) -> Dict[str, float]:
    w = ctx.dt.weekday() if ctx.dt is not None else 0
    return {f"Weekday_{i}": 1.0 if w == i else 0.0 for i in range(7)}


def _extract_role_occupancy(ctx: FeatureContext) -> Dict[str, float]:
    return {f"Role_{i}_OC": float(val) for i, val in enumerate(ctx.role_oc)}


def _extract_prev_duration(ctx: FeatureContext) -> Dict[str, float]:
    return {"Prev_Proc_Time": float(ctx.prev_dur)}


def _extract_case_attributes(ctx: FeatureContext) -> Dict[str, Any]:
    res = {}
    if ctx.case_attrs:
        for k, v in ctx.case_attrs.items():
            try:
                res[k] = float(v)
            except (ValueError, TypeError):
                res[k] = str(v) if v is not None else "UNKNOWN"
    else:
        res["RequestedAmount"] = float(ctx.requested_amount)
    return res


def _extract_event_attributes(ctx: FeatureContext) -> Dict[str, Any]:
    res = {}
    if ctx.event_attrs:
        for k, v in ctx.event_attrs.items():
            try:
                res[k] = float(v)
            except (ValueError, TypeError):
                res[k] = str(v) if v is not None else "UNKNOWN"
    return res


def _extract_resource_id(ctx: FeatureContext) -> Dict[str, str]:
    return {"Resource_ID": str(ctx.worker_id) if ctx.worker_id else "UNKNOWN"}


def _extract_trace_history(ctx: FeatureContext) -> Dict[str, float]:
    res = {}
    for act in ctx.tracked_activities:
        clean_name = act.replace(" ", "_").replace("-", "_")
        cnt = ctx.trace_history.get(act, 0)
        res[f"has_{clean_name}"] = 1.0 if cnt > 0 else 0.0
        if getattr(ctx, "include_counts", False):
            res[f"count_{clean_name}"] = float(cnt)
    return res


# Central Registry of modular feature extractors
FEATURE_REGISTRY: Dict[str, Callable[[FeatureContext], Dict[str, Any]]] = {
    "wip": _extract_wip,
    "ac_wip": _extract_ac_wip,
    "daytime": _extract_daytime,
    "weekday": _extract_weekday,
    "role_occupancy": _extract_role_occupancy,
    "prev_duration": _extract_prev_duration,
    "trace_history": _extract_trace_history,
    "case_attributes": _extract_case_attributes,
    "event_attributes": _extract_event_attributes,
    "resource_id": _extract_resource_id,
}


class FeatureManager:
    """
    Manages active feature extractors configured via YAML.
    Provides backwards compatibility with legacy universal toggles while enabling
    explicit feature selection via `selected_features: [...]`.
    """
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        feat_cfg = self.cfg.get("features", {})

        # Priority 1: Explicit selected_features list in YAML
        if "selected_features" in feat_cfg and feat_cfg["selected_features"]:
            self.selected_features = [str(f).lower() for f in feat_cfg["selected_features"]]
            if feat_cfg.get("case_attributes", None) == [] and "case_attributes" in self.selected_features:
                self.selected_features.remove("case_attributes")
            if feat_cfg.get("event_attributes", None) == [] and "event_attributes" in self.selected_features:
                self.selected_features.remove("event_attributes")
        else:
            # Priority 2: Backwards compatibility mapping from legacy config structure
            univ = feat_cfg.get("universal", {})
            selected = []
            if univ.get("use_wip", True):
                selected.append("wip")
            if univ.get("use_ac_wip", True):
                selected.append("ac_wip")
            if univ.get("use_calendar_time", True):
                selected.extend(["daytime", "weekday"])
            if univ.get("use_role_occupancy", True):
                selected.append("role_occupancy")
            if univ.get("use_prev_duration", True):
                selected.append("prev_duration")

            if feat_cfg.get("case_attributes", None) != []:
                selected.append("case_attributes")
            if feat_cfg.get("event_attributes", None):
                selected.append("event_attributes")

            # Check if resource is requested in event_attributes
            ev_attrs = [str(x).lower() for x in feat_cfg.get("event_attributes", [])]
            if "org:resource" in ev_attrs or "resource_id" in ev_attrs:
                selected.append("resource_id")

            self.selected_features = selected

        self.include_counts = bool(feat_cfg.get("trace_history", {}).get("include_counts", False))

    def build_features(self, ctx: FeatureContext) -> Dict[str, Any]:
        """Runs all active feature extractors on the provided context."""
        if not getattr(ctx, "include_counts", False) and self.include_counts:
            ctx.include_counts = True

        feats: Dict[str, Any] = {}
        for f_name in self.selected_features:
            extractor = FEATURE_REGISTRY.get(f_name)
            if extractor is not None:
                feats.update(extractor(ctx))
        return feats

    def is_feature_active(self, feature_name: str) -> bool:
        """Checks if a feature is enabled in the configuration."""
        return feature_name.lower() in self.selected_features

