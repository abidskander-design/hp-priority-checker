"""OpenStack release calendar and proximity features."""

from __future__ import annotations

from datetime import date, datetime

# Confirmed OpenStack named releases with their GA dates
OPENSTACK_RELEASES: list[tuple[str, date]] = [
    ("Yoga",      date(2022, 3, 30)),
    ("Zed",       date(2022, 10, 5)),
    ("Antelope",  date(2023, 3, 22)),
    ("Bobcat",    date(2023, 10, 4)),
    ("Caracal",   date(2024, 4, 3)),
    ("Dalmatian", date(2024, 10, 2)),
    ("Epoxy",     date(2025, 4, 2)),
    ("Flamingo",  date(2025, 10, 1)),
]


def days_to_nearest_release(ts: datetime | date | str) -> int:
    """Return the absolute number of days between *ts* and the nearest release.

    A negative value means *ts* is before the release (upcoming); positive
    means it is after.  We return the *minimum absolute* distance so the
    feature captures proximity regardless of direction.
    """
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, datetime):
        d = ts.date()
    else:
        d = ts

    release_dates = [r[1] for r in OPENSTACK_RELEASES]
    return min(abs((d - rd).days) for rd in release_dates)


def days_until_next_release(ts: datetime | date | str) -> int:
    """Return days until the *next* upcoming release from *ts* (negative if past all)."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, datetime):
        d = ts.date()
    else:
        d = ts

    upcoming = [(rd - d).days for _, rd in OPENSTACK_RELEASES if rd >= d]
    return min(upcoming) if upcoming else -1


def nearest_release_name(ts: datetime | date | str) -> str:
    """Return the name of the nearest release to *ts*."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, datetime):
        d = ts.date()
    else:
        d = ts

    name, _ = min(OPENSTACK_RELEASES, key=lambda r: abs((d - r[1]).days))
    return name
