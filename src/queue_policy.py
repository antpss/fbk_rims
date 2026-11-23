"""Validated task scheduling policies; priority values never belong to ML payloads."""
import math
from numbers import Real


class QueuePolicy:
    def __init__(self, config=None):
        config = config or {}
        self.type = config.get("type", "fifo")
        if self.type not in ("fifo", "lifo", "priority"):
            raise ValueError("simulation.queue.type must be fifo, lifo or priority")
        if config.get("tie_breaker", "fifo") != "fifo":
            raise ValueError("simulation.queue.tie_breaker must be fifo")
        self.priority_config = config.get("priority", {})
        self.attribute = self.priority_config.get("attribute", "Priority")
        if not isinstance(self.attribute, str) or not self.attribute or self.attribute.startswith("case:") or self.attribute == "concept:name":
            raise ValueError("queue priority attribute must be a nonempty name without 'case:' and cannot be 'concept:name'")
        if self.type != "priority":
            return
        p = self.priority_config
        self.levels = p.get("levels")
        if type(self.levels) is not int or self.levels < 1:
            raise ValueError("queue priority levels must be a positive integer")
        self.source = p.get("source_attribute", self.attribute)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("queue priority source_attribute must be a nonempty name")
        self.on_missing = p.get("on_missing", "error")
        if self.on_missing not in ("error", "default"):
            raise ValueError("queue priority on_missing must be error or default")
        self.default = p.get("default_priority")
        if self.on_missing == "default":
            self._check_priority(self.default, configuration=True)
        self.mapping = p.get("mapping")
        self.ranges = []
        if self.mapping is None:
            return
        kind = self.mapping.get("kind")
        if kind == "categorical":
            values = self.mapping.get("values")
            if not isinstance(values, dict) or not values:
                raise ValueError("categorical priority mapping needs nonempty values")
            for value in values.values():
                self._check_priority(value, configuration=True)
        elif kind == "numeric_ranges":
            ranges = self.mapping.get("ranges")
            if not isinstance(ranges, list) or not ranges:
                raise ValueError("numeric priority mapping needs nonempty ranges")
            for entry in ranges:
                lo = self._bound(entry, "min_inclusive", -math.inf)
                hi = self._bound(entry, "max_exclusive", math.inf)
                if lo >= hi:
                    raise ValueError("priority range lower bound must be less than upper bound")
                self.ranges.append((lo, hi, self._check_priority(entry.get("priority"), configuration=True)))
            self.ranges.sort()
            for left, right in zip(self.ranges, self.ranges[1:]):
                if left[1] > right[0]:
                    raise ValueError("priority ranges must not overlap")
            # Gaps are allowed, but an encountered uncovered value always fails explicitly.
        else:
            raise ValueError("priority mapping kind must be categorical or numeric_ranges")

    @staticmethod
    def _bound(entry, key, default):
        if key not in entry:
            return default
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"priority range {key} must be a finite number")
        return float(value)

    def _check_priority(self, value, configuration=False):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or int(value) != value
                or not 1 <= value <= self.levels
                or (configuration and type(value) is not int)):
            raise ValueError(f"priority must be an integer between 1 and {self.levels}; got {value!r}")
        return int(value)

    def resolve(self, value, case_id=None):
        """Map an original, unimputed case value to scheduling-only priority."""
        try:
            if value is None or (isinstance(value, Real) and math.isnan(value)):
                if self.on_missing == "default":
                    return self.default
                raise ValueError(f"missing source attribute {self.source!r}")
            if self.mapping is None:
                return self._check_priority(value)
            if self.mapping["kind"] == "categorical":
                try:
                    return self.mapping["values"][value]
                except (KeyError, TypeError):
                    raise ValueError(f"unmapped priority category {value!r}") from None
            if isinstance(value, bool):
                raise ValueError(f"invalid numeric priority source {value!r}")
            try:
                number = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"invalid numeric priority source {value!r}") from None
            if not math.isfinite(number):
                raise ValueError(f"invalid numeric priority source {value!r}")
            for lo, hi, priority in self.ranges:
                if lo <= number < hi:
                    return priority
            raise ValueError(f"uncovered priority source value {value!r}")
        except ValueError as exc:
            raise ValueError(f"Queue priority for case {case_id!r}: {exc}") from exc

    def sort_key(self, request):
        seq = request["sequence"]
        if self.type == "lifo":
            return (-seq,)
        if self.type == "priority":
            return (request["priority"], seq)
        return (seq,)

    def source_column(self, cfg):
        """Accept either a configured ML feature alias or an independent log column."""
        for attr in cfg.get("features", {}).get("case_attributes", []):
            if attr.get("feature_name") == self.source:
                return attr["column"]
        return self.source

    def validate_features(self, cfg):
        if self.type != "priority":
            return
        for group in ("case_attributes", "event_attributes"):
            for attr in cfg.get("features", {}).get(group, []):
                names = (attr.get("column", ""), attr.get("feature_name", "")) if isinstance(attr, dict) else (str(attr),)
                if any(name.removeprefix("case:") == self.attribute for name in names):
                    raise ValueError(f"Scheduling attribute {self.attribute!r} must not be configured as an ML feature")
