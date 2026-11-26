"""NOAA CPC ENSO Diagnostic Discussion: alert status, issue date and synopsis.

ENSO is a tropical-Pacific signal. Its local effect (say, on Indian monsoon rain)
is reasoned about by the model with regional evidence; it is never turned into
a demand number directly here.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from html import unescape

from engine.reason.connectors.http import HttpRequest, ProviderError, get_ok
from engine.reason.schemas import Signal

NOAA_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml"


def page_text(html: str) -> str:
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    text = unescape(re.sub(r"<[^>]+>", " ", body))
    return " ".join(text.replace("�", "n").split())


def parse(html: str) -> dict:
    text = page_text(html)
    issued = re.search(r"issued by .{0,80}?(\d{1,2} [A-Z][a-z]+ 20\d{2})", text)
    if issued is None:
        raise ProviderError("NOAA page did not contain an issued diagnostic discussion")
    issue_date = datetime.strptime(issued.group(1), "%d %B %Y").date()
    status_match = re.search(r"ENSO Alert System Status:\s*(.+?)\s+Synopsis:", text)
    after = text.split("Synopsis:", 1)[1][:800] if "Synopsis:" in text else ""
    synopsis = re.split(r"(?<=[.)])\s+(?=[A-Z])", after.strip())
    return {
        "issue_date": issue_date.isoformat(),
        "status": status_match.group(1).strip() if status_match else "not stated",
        "synopsis": " ".join(synopsis[:2])[:400],
    }


def fetch(driver, ctx) -> list[Signal]:
    if not ctx.sources.enso:
        return []
    response = get_ok(ctx.http, HttpRequest(NOAA_URL, headers={"Accept": "text/html"}), 30)
    data = parse(response.text())
    if date.fromisoformat(data["issue_date"]) > ctx.as_of.date():
        raise ProviderError("NOAA discussion is dated after the review date")
    return [
        Signal(
            driver_id=driver.id,
            kind="enso",
            summary=(
                f"NOAA CPC ENSO discussion issued {data['issue_date']}: status "
                f"'{data['status']}'. {data['synopsis']}"
            ),
            data=data,
            source_ref=NOAA_URL,
        )
    ]
