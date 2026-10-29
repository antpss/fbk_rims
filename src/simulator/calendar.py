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
        """Calculates seconds until the next working shift begins (O(1) closed-form calculation)."""
        if not self.enabled or self.is_working_hour(dt):
            return 0.0

        if dt.weekday() in self.work_days and dt.hour < self.hour_start:
            next_start = dt.replace(hour=self.hour_start, minute=0, second=0, microsecond=0)
            return (next_start - dt).total_seconds()

        # After shift or during non-working day: jump to next scheduled morning in O(1)
        w = dt.weekday()
        days_ahead = min(((d - w - 1) % 7) + 1 for d in self.work_days)
        next_start = (dt + timedelta(days=days_ahead)).replace(hour=self.hour_start, minute=0, second=0, microsecond=0)
        return (next_start - dt).total_seconds()

    def calculate_calendar_duration(self, sim_seconds, duration_seconds, role_or_act=None):
        """
        Advances working duration across nights and weekends using closed-form week/shift arithmetic.
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

        if self.hour_end >= 24:
            day_end = (curr_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            day_end = curr_dt.replace(hour=self.hour_end, minute=0, second=0, microsecond=0)
        today_window = max(0.0, (day_end - curr_dt).total_seconds())

        if duration_seconds <= today_window:
            curr_dt += timedelta(seconds=duration_seconds)
            return max(duration_seconds, (curr_dt - start_dt).total_seconds())

        work_left = duration_seconds - today_window
        curr_dt = day_end

        # Jump to next working morning
        w = curr_dt.weekday()
        days_ahead = min(((d - w - 1) % 7) + 1 for d in self.work_days)
        curr_dt = (curr_dt + timedelta(days=days_ahead)).replace(hour=self.hour_start, minute=0, second=0, microsecond=0)

        shift_len = (self.hour_end - self.hour_start) * 3600.0
        full_shifts, rem_seconds = divmod(work_left, shift_len)
        if rem_seconds == 0.0 and full_shifts > 0:
            full_shifts -= 1
            rem_seconds = shift_len

        full_weeks, extra_shifts = divmod(int(full_shifts), len(self.work_days))

        # O(1) jump across full weeks (7 calendar days per len(work_days) work shifts)
        if full_weeks > 0:
            curr_dt += timedelta(days=full_weeks * 7)

        # Step through remaining extra shifts (at most len(work_days) - 1 days)
        for _ in range(extra_shifts):
            w = curr_dt.weekday()
            days_ahead = min(((d - w - 1) % 7) + 1 for d in self.work_days)
            curr_dt = (curr_dt + timedelta(days=days_ahead)).replace(hour=self.hour_start, minute=0, second=0, microsecond=0)

        curr_dt += timedelta(seconds=rem_seconds)
        return max(duration_seconds, (curr_dt - start_dt).total_seconds())

    def calculate_net_working_seconds(self, start_dt, end_dt, role_or_act=None, allow_zero=False):
        """
        Computes effective business working duration (excluding nights and weekends) with full-week chunking.
        Uses specialized per-role calendar if defined, otherwise falls back to global calendar.
        If allow_zero is True (e.g. for waiting times), zero net seconds across off-shift
        hours is returned without applying duration-specific overtime fallbacks.
        """
        if role_or_act and role_or_act in self.role_calendars:
            return self.role_calendars[role_or_act].calculate_net_working_seconds(start_dt, end_dt, allow_zero=allow_zero)

        s_dt = start_dt.replace(tzinfo=None) if hasattr(start_dt, "tzinfo") and start_dt.tzinfo else start_dt
        e_dt = end_dt.replace(tzinfo=None) if hasattr(end_dt, "tzinfo") and end_dt.tzinfo else end_dt

        raw_seconds = max(0.0, (e_dt - s_dt).total_seconds())
        if not self.enabled or raw_seconds <= 0:
            return raw_seconds

        shift_len = (self.hour_end - self.hour_start) * 3600.0

        if s_dt.date() == e_dt.date():
            if s_dt.date().weekday() in self.work_days:
                shift_s = datetime(s_dt.year, s_dt.month, s_dt.day, self.hour_start, 0, 0)
                if self.hour_end >= 24:
                    shift_e = datetime(s_dt.year, s_dt.month, s_dt.day, 0, 0, 0) + timedelta(days=1)
                else:
                    shift_e = datetime(s_dt.year, s_dt.month, s_dt.day, self.hour_end, 0, 0)
                win_s = max(s_dt, shift_s)
                win_e = min(e_dt, shift_e)
                net_seconds = max(0.0, (win_e - win_s).total_seconds()) if win_e > win_s else 0.0
            else:
                net_seconds = 0.0
        else:
            net_seconds = 0.0
            # 1. Day 1 (start day)
            if s_dt.date().weekday() in self.work_days:
                shift_s = datetime(s_dt.year, s_dt.month, s_dt.day, self.hour_start, 0, 0)
                if self.hour_end >= 24:
                    shift_e = datetime(s_dt.year, s_dt.month, s_dt.day, 0, 0, 0) + timedelta(days=1)
                else:
                    shift_e = datetime(s_dt.year, s_dt.month, s_dt.day, self.hour_end, 0, 0)
                win_s = max(s_dt, shift_s)
                if shift_e > win_s:
                    net_seconds += (shift_e - win_s).total_seconds()

            # 2. Intermediate days with O(1) full-week chunking
            d_start = s_dt.date() + timedelta(days=1)
            d_end = e_dt.date() - timedelta(days=1)
            if d_end >= d_start:
                total_inter_days = (d_end - d_start).days + 1
                full_weeks, rem_days = divmod(total_inter_days, 7)
                net_seconds += full_weeks * (len(self.work_days) * shift_len)

                curr_d = d_start + timedelta(days=full_weeks * 7)
                while curr_d <= d_end:
                    if curr_d.weekday() in self.work_days:
                        net_seconds += shift_len
                    curr_d += timedelta(days=1)

            # 3. Final Day (end day)
            if e_dt.date().weekday() in self.work_days:
                shift_s = datetime(e_dt.year, e_dt.month, e_dt.day, self.hour_start, 0, 0)
                win_e = min(e_dt, shift_s + timedelta(seconds=shift_len))
                if win_e > shift_s:
                    net_seconds += (win_e - shift_s).total_seconds()

        if allow_zero:
            return net_seconds

        # If an activity starts on a scheduled working day and finishes on the same calendar day
        # (e.g. worker stayed late past shift end to complete an active call or application),
        # credit the continuous single-session working duration rather than truncating to near-zero.
        if s_dt.date() == e_dt.date() and s_dt.date().weekday() in self.work_days and raw_seconds <= 4 * 3600.0:
            net_seconds = max(net_seconds, raw_seconds)
        elif net_seconds <= 0:
            net_seconds = min(raw_seconds, 3600.0)

        return net_seconds

