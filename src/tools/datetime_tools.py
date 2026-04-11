"""Date/time tools so the agent always knows the real current time."""
from __future__ import annotations

from datetime import datetime, timezone


def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO-8601 string.

    Use this whenever you need to know today's date, the current time, or
    want to timestamp something accurately.
    """
    now = datetime.now(timezone.utc)
    return (
        f"Current UTC datetime: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"ISO-8601: {now.isoformat()}\n"
        f"Day of week: {now.strftime('%A')}\n"
        f"Unix timestamp: {int(now.timestamp())}"
    )


def get_timezone_time(tz_name: str) -> str:
    """Return the current time in a given IANA timezone (e.g. 'America/New_York').

    Args:
        tz_name: IANA timezone identifier such as 'America/New_York',
                 'Europe/London', 'Asia/Tokyo', 'US/Eastern', etc.
    """
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        return (
            f"Current time in {tz_name}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"UTC offset: {now.strftime('%z')}\n"
            f"Day of week: {now.strftime('%A')}"
        )
    except Exception as exc:
        return f"Error: could not get time for timezone '{tz_name}': {exc}"
