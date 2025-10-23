import hashlib
from datetime import datetime, timezone
from dateutil import parser as dtparser


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def parse_pub_date(entry) -> datetime:
    for k in ("published", "updated"):
        if getattr(entry, k, None):
            try:
                return dtparser.parse(getattr(entry, k)).astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)
