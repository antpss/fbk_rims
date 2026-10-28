#!/usr/bin/env python3
"""
src/simulator/calendar.py

Operational work shifts, night closures, and weekend stops.
Supports global business hours as well as per-role / per-activity calendar overrides.
Ensures resources only perform work during configured operational windows.
"""

from datetime import datetime, timedelta


class CalendarManager:
    """
    Controls operational work shifts, night closures, and weekend stops.
    Supports a global business shift as well as per-role / per-activity calendar overrides.
    Ensures human resources only perform work during configured office hours.
    """
    def __init__(self, enabled=True, work_days=(0, 1, 2, 3, 4), hour_start=8, hour_end=20, ref_dt=None, per_role=None):
        self.enabled = enabled
        self.work_days = set(work_days)
        self.hour_start = hour_start
        self.hour_end = hour_end
        self.ref_dt = ref_dt or datetime(2011, 10, 1, 0, 0, 0)
        self.per_role_cfgs = per_role or {}
        self.role_calendars = {}

        # Instantiate specialized per-role / per-activity calendars
        for role_name, r_cfg in self.per_role_cfgs.items():
            self.role_calendars[role_name] = CalendarManager(
                enabled=r_cfg.get("enabled", self.enabled),
                work_days=r_cfg.get("work_days", list(self.work_days)),
                hour_start=r_cfg.get("hour_start", self.hour_start),
                hour_end=r_cfg.get("hour_end", self.hour_end),
                ref_dt=self.ref_dt
            )

    @classmethod
    def from_config(cls, cfg, ref_dt=None):
        """Factory method constructing CalendarManager directly from central config.yaml dictionary."""
        cal_cfg = cfg.get("simulation", {}).get("calendar", {})
        return cls(
            enabled=cal_cfg.get("enabled", True),
            work_days=cal_cfg.get("work_days", [0, 1, 2, 3, 4]),
            hour_start=cal_cfg.get("hour_start", 8),
            hour_end=cal_cfg.get("hour_end", 20),
            ref_dt=ref_dt,
            per_role=cal_cfg.get("per_role", {})
        )

    def set_ref_dt(self, ref_dt):
        """Synchronizes reference start timestamp across global and per-role calendars."""
        self.ref_dt = ref_dt
        for rc in self.role_calendars.values():
            rc.set_ref_dt(ref_dt)

    def get_calendar(self, role_or_act=None):
        """Returns specialized calendar if configured for role/activity, otherwise global calendar."""
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act]
        return self

    def to_datetime(self, sim_seconds):
        """Converts simulation seconds elapsed from ref_dt to a real datetime object."""
        return self.ref_dt + timedelta(seconds=sim_seconds)

    def is_working_hour(self, dt):
        """Checks if a given datetime falls within configured working days and office hours."""
        if not self.enabled:
            return True
        if dt.weekday() not in self.work_days:
            return False
        return self.hour_start <= dt.hour < self.hour_end

    def get_gap_to_next_work_window(self, dt):
        """Calculates seconds until the next working shift begins."""
        if not self.enabled or self.is_working_hour(dt):
            return 0.0

        curr = dt
        while True:
            if curr.weekday() not in self.work_days:
                curr = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
                if curr.weekday() in self.work_days:
                    return (curr - dt).total_seconds()
            elif curr.hour < self.hour_start:
                next_start = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0)
                return (next_start - dt).total_seconds()
            elif curr.hour >= self.hour_end:
                next_day = curr.replace(hour=self.hour_start, minute=0, second=0, microsecond=0) + timedelta(days=1)
                curr = next_day
                if curr.weekday() in self.work_days:
                    return (curr - dt).total_seconds()
            else:
                return max(0.0, (curr - dt).total_seconds())

    def calculate_calendar_duration(self, sim_seconds, duration_seconds, role_or_act=None):
        """
        Advances working duration across nights and weekends.
        Uses specialized per-role calendar if defined, otherwise falls back to global calendar.
        Returns total elapsed wall-clock seconds in the simulation.
        """
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act].calculate_calendar_duration(sim_seconds, duration_seconds)

        if not self.enabled or duration_seconds <= 0:
            return max(0.0, duration_seconds)

        start_dt = self.to_datetime(sim_seconds)
        initial_pause = self.get_gap_to_next_work_window(start_dt)
        curr_dt = start_dt + timedelta(seconds=initial_pause)

        work_left = duration_seconds
        while work_left > 0:
            if self.hour_end >= 24:
                day_end = (curr_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                day_end = curr_dt.replace(hour=self.hour_end, minute=0, second=0, microsecond=0)
            window_seconds = max(0.0, (day_end - curr_dt).total_seconds())

            if work_left <= window_seconds:
                curr_dt += timedelta(seconds=work_left)
                work_left = 0
            else:
                work_left -= window_seconds
                curr_dt = day_end
                gap = self.get_gap_to_next_work_window(curr_dt)
                curr_dt += timedelta(seconds=gap)

        total_elapsed = max(duration_seconds, (curr_dt - start_dt).total_seconds())
        return total_elapsed

    def calculate_net_working_seconds(self, start_dt, end_dt, role_or_act=None):
        """
        Computes effective business working duration (excluding nights and weekends).
        Uses specialized per-role calendar if defined, otherwise falls back to global calendar.
        """
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act].calculate_net_working_seconds(start_dt, end_dt)

        s_dt = start_dt.replace(tzinfo=None) if hasattr(start_dt, "tzinfo") and start_dt.tzinfo else start_dt
        e_dt = end_dt.replace(tzinfo=None) if hasattr(end_dt, "tzinfo") and end_dt.tzinfo else end_dt

        raw_seconds = max(0.0, (e_dt - s_dt).total_seconds())
        if not self.enabled or raw_seconds <= 0:
            return raw_seconds

        curr_d = s_dt.date()
        net_seconds = 0.0
        while curr_d <= e_dt.date():
            if curr_d.weekday() in self.work_days:
                shift_s = datetime(curr_d.year, curr_d.month, curr_d.day, self.hour_start, 0, 0)
                if self.hour_end >= 24:
                    shift_e = datetime(curr_d.year, curr_d.month, curr_d.day, 0, 0, 0) + timedelta(days=1)
                else:
                    shift_e = datetime(curr_d.year, curr_d.month, curr_d.day, self.hour_end, 0, 0)
                win_s = max(s_dt, shift_s)
                win_e = min(e_dt, shift_e)
                if win_e > win_s:
                    net_seconds += (win_e - win_s).total_seconds()
            curr_d += timedelta(days=1)

        # If an activity starts on a scheduled working day and finishes on the same calendar day
        # (e.g. worker stayed late past shift end to complete an active call or application),
        # credit the continuous single-session working duration rather than truncating to near-zero.
        if s_dt.date() == e_dt.date() and s_dt.date().weekday() in self.work_days and raw_seconds <= 4 * 3600.0:
            net_seconds = max(net_seconds, raw_seconds)
        elif net_seconds <= 0:
            net_seconds = min(raw_seconds, 3600.0)

        return net_seconds

