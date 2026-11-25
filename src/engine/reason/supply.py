"""Normalize ERP operating evidence and derive dated supply exposures."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from engine.reason.config import OPERATIONAL_FIELDS, OperationalSource
from engine.reason.erp import SourceRow
from engine.reason.facts import local_midnight, required_text, scalar_value, source_time
from engine.reason.providers import ParsedEvidence, evidence_revision
from engine.reason.schemas import (
    EntityRef,
    EvidenceRevision,
    ExposureGroup,
    ExposureMember,
    ForecastRow,
    ForecastSnapshot,
    OperationalRecord,
    QueryScope,
    content_digest,
)

_MOVEMENT = {
    "usable_stock": "snapshot",
    "purchase_order": "inflow",
    "receipt": "inflow",
    "consumption": "outflow",
}
_STATUSES = {
    "purchase_order": {"open", "confirmed", "contingent", "cancelled"},
    "receipt": {"confirmed", "received", "contingent", "cancelled"},
    "alternate": {"approved", "confirmed", "contingent"},
    "allocation": {"approved", "confirmed"},
    "order_amendment": {"confirmed", "contingent", "cancelled"},
    "contract_option": {"confirmed", "contingent", "cancelled"},
    "method_parameter": {"approved", "confirmed"},
    "planner_assumption": {"approved", "confirmed"},
}
_INACTIVE = {"withdrawn", "cancelled"}


def _present(value: object) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _normalize_row(
    source: OperationalSource,
    scope: QueryScope,
    raw: SourceRow,
    captured_at: datetime,
) -> EvidenceRevision:
    row = {key: val for key, val in raw.items() if _present(val)}
    dataset = source.dataset
    zone = ZoneInfo(dict(scope.filters)["timezone"])
    start = source_time(row["effective_from"], zone, source)
    end = source_time(row["effective_to"], zone, source) if "effective_to" in row else None
    known = source_time(row["known_at"], zone, source, known=True) if "known_at" in row else None
    if end is not None and end <= start:
        raise ValueError("Invalid effective interval")
    status = required_text(row.get("status", "confirmed"), "status").lower()
    if status not in _STATUSES.get(dataset, {"active", "confirmed"}) | _INACTIVE:
        raise ValueError("Invalid operational status")
    keys = OPERATIONAL_FIELDS[dataset]
    if status not in _INACTIVE and keys - row.keys():
        raise ValueError("Incomplete operating record")
    fields = {
        key: required_text(row[key], key)
        for key in OperationalRecord.model_fields.keys() & row.keys()
        if key
        not in {
            "status",
            "lead_time_days",
            "conversion_factor",
            "ordered_at",
            "original_date",
            "baseline_includes",
            "training_cutoff",
            "minimum_driver",
            "maximum_driver",
            "lag_months",
            "min_horizon",
            "max_horizon",
        }
    }
    for key in ("ordered_at", "original_date"):
        if key in row:
            fields[key] = source_time(row[key], zone, source)
    if "lead_time_days" in row:
        lead = scalar_value(row["lead_time_days"], "number")
        if lead < 0 or lead > 3660 or not lead.is_integer():
            raise ValueError("Lead time requires a whole number of days")
        fields["lead_time_days"] = int(lead)
    if "baseline_includes" in row:
        fields["baseline_includes"] = scalar_value(row["baseline_includes"], "boolean")
    if "training_cutoff" in row:
        fields["training_cutoff"] = source_time(row["training_cutoff"], zone, source)
    for key in ("minimum_driver", "maximum_driver", "lag_months", "min_horizon", "max_horizon"):
        if key in row:
            value = scalar_value(row[key], "number")
            if key in {"lag_months", "min_horizon", "max_horizon"}:
                if not value.is_integer():
                    raise ValueError(f"{key} requires a whole number")
                value = int(value)
            fields[key] = value
    if (
        status == "contingent"
        and dataset in {"order_amendment", "contract_option"}
        and "condition" not in row
    ):
        raise ValueError("A contingent order requires a condition")
    if ("converted_unit" in row) != ("conversion_factor" in row):
        raise ValueError("Unit conversion requires a unit and factor")
    if "conversion_factor" in row:
        fields["conversion_factor"] = scalar_value(row["conversion_factor"], "number")
        if fields["conversion_factor"] <= 0:
            raise ValueError("Invalid unit conversion")
    quantity = scalar_value(row["quantity"], "number") if "quantity" in row else None
    if quantity is not None and (dataset in _MATERIAL_QUANTITIES and quantity < 0):
        raise ValueError("Supply quantities must be nonnegative")
    if dataset == "bom" and quantity == 0:
        raise ValueError("BOM requires positive quantity per finished unit")
    if (
        dataset == "usable_stock"
        and source.stock_basis == "warehouse"
        and "warehouse_id" not in row
    ):
        raise ValueError("Warehouse stock requires warehouse ID")
    raw_date = row["effective_from"]
    coarse = isinstance(raw_date, date) and not isinstance(raw_date, datetime)
    if isinstance(raw_date, str):
        coarse = len(raw_date.strip()) == 10
    record_id = required_text(row["record_id"], "record_id")
    record = OperationalRecord(
        dataset=dataset,
        status=status,
        movement=_MOVEMENT.get(dataset, "reference"),
        stock_basis=source.stock_basis if dataset == "usable_stock" else None,
        date_precision="day" if coarse else "instant",
        **fields,
    )
    return evidence_revision(
        scope.client_id,
        "erp",
        source.id,
        record.receipt_id
        if dataset in {"receipt", "purchase_order"} and record.receipt_id
        else record_id,
        captured_at,
        released_at=known,
        source_known_at=known,
        source_event_id=required_text(row["event_id"], "event_id") if "event_id" in row else None,
        source_ref=f"{source.kind}:{source.relation or source.path}#{record_id}",
        source_version=required_text(row["version"], "version") if "version" in row else None,
        valid_from=start,
        valid_to=end,
        location_id=record.site_id,
        quantity=quantity,
        unit=required_text(row["unit"], "unit") if "unit" in row else None,
        operational=record,
    )


_MATERIAL_QUANTITIES = {
    "usable_stock",
    "purchase_order",
    "receipt",
    "consumption",
    "alternate",
    "allocation",
    "bom",
}


def parse_operational(
    source: OperationalSource,
    scope: QueryScope,
    rows: list[SourceRow],
    truncated: bool,
    captured_at: datetime,
) -> ParsedEvidence:
    revisions = {}
    invalid = 0
    for row in rows:
        try:
            revision = _normalize_row(source, scope, row, captured_at)
        except (ValueError, TypeError, KeyError, OverflowError):
            invalid += 1
            continue
        revisions.setdefault(revision.evidence_id, revision)
    # Sort explicit source releases chronologically before assigning stored revision numbers.
    return ParsedEvidence(
        tuple(sorted(revisions.values(), key=lambda item: item.source_known_at or captured_at)),
        coverage="partial" if truncated or invalid else "complete",
        reason="erp_row_limit_or_invalid_rows" if truncated or invalid else None,
        covered_from=scope.valid_from if not truncated and not invalid else None,
        covered_to=scope.valid_to if not truncated and not invalid else None,
    )


def active_operational(evidence: tuple[EvidenceRevision, ...]) -> tuple[EvidenceRevision, ...]:
    latest = {}
    for item in sorted(
        (item for item in evidence if item.operational),
        key=lambda e: (e.source_known_at or e.captured_at, e.revision, e.evidence_id),
    ):
        latest[item.record_id] = item
    invalid_receipts = {}
    for item in latest.values():
        record = item.operational
        if record.dataset == "receipt" and record.receipt_id:
            invalid_receipts.setdefault(record.receipt_id, []).append(item)
    return tuple(
        item
        for item in latest.values()
        if item.operational.status not in _INACTIVE
        and not (
            item.operational.dataset == "purchase_order"
            and item.operational.receipt_id in invalid_receipts
        )
    )


def member(row: ForecastRow, shortfall: float | None = None) -> ExposureMember:
    return ExposureMember(
        series_id=row.series_id,
        period=row.period,
        item_id=row.item_id,
        customer_id=row.customer_id,
        shortfall=shortfall,
    )


def _in_period(record: EvidenceRevision, start: datetime, end: datetime) -> bool:
    return record.valid_from < end and (record.valid_to is None or record.valid_to > start)


def _matches_item(evidence: EvidenceRevision, row: ForecastRow) -> bool:
    record = evidence.operational
    return (
        record.item_id == row.item_id
        and record.customer_id in (None, row.customer_id)
        and record.series_id in (None, row.series_id)
    )


def _dependency_links(records: list[EvidenceRevision], row: ForecastRow) -> list[EvidenceRevision]:
    matching = [
        link
        for link in records
        if link.operational.dataset in {"bom", "vendor_link"}
        and link.operational.item_id is not None
        and _matches_item(link, row)
        and _in_period(link, row.target_start, row.target_end)
    ]
    bom = [link for link in matching if link.operational.dataset == "bom"]
    return bom or matching


def _amount(record: EvidenceRevision, unit: str) -> Decimal | None:
    if record.quantity is None:
        return None
    amount = Decimal(str(record.quantity))
    if record.unit == unit:
        return amount
    value = record.operational
    if value.converted_unit == unit and value.conversion_factor is not None:
        return amount * Decimal(str(value.conversion_factor))
    return None


def _balance(
    records: list[EvidenceRevision],
    start: datetime,
    end: datetime,
    zone: ZoneInfo,
) -> tuple[Decimal | None, datetime | None, str | None, set[str], set[str]]:
    stocks = [
        record
        for record in records
        if record.operational.dataset == "usable_stock"
        and record.valid_from <= start
        and (record.valid_to is None or record.valid_to > start)
    ]
    flags, evidence = set(), set()
    if not stocks:
        later = [
            record
            for record in records
            if record.operational.dataset == "usable_stock" and start < record.valid_from < end
        ]
        if not later:
            return None, None, None, {"missing_usable_stock"}, evidence
        first = min(record.valid_from for record in later)
        stocks = [record for record in later if record.valid_from == first]
        flags.add("no_opening_stock_snapshot")
    site_totals = [stock for stock in stocks if stock.operational.stock_basis == "site_total"]
    if site_totals:
        stocks = site_totals
    elif len({stock.operational.warehouse_id for stock in stocks}) != 1 or any(
        record.operational.movement in {"inflow", "outflow"}
        and record.operational.warehouse_id != stocks[0].operational.warehouse_id
        and stocks[0].valid_from <= record.valid_from < end
        for record in records
    ):
        return None, None, None, {"unverified_warehouse_transfers"}, {r.evidence_id for r in stocks}
    latest = max(stock.valid_from for stock in stocks)
    selected = [stock for stock in stocks if stock.valid_from == latest]
    if len(selected) != 1:
        return None, None, None, {"ambiguous_stock_baseline"}, {r.evidence_id for r in selected}
    stock = selected[0]
    unit = stock.unit
    evidence.add(stock.evidence_id)
    movements = [
        record
        for record in records
        if latest <= record.valid_from < end
        and record.operational.movement in {"inflow", "outflow"}
    ]
    # Two provider views claiming the same receipt identity are unresolved, not additive.
    receipt_ids = [
        r.operational.receipt_id
        for r in movements
        if r.operational.movement == "inflow" and r.operational.receipt_id
    ]
    if len(receipt_ids) != len(set(receipt_ids)):
        return (
            None,
            None,
            unit,
            {"ambiguous_receipt_identity"},
            evidence | {r.evidence_id for r in movements},
        )
    by_day = defaultdict(list)
    for record in movements:
        by_day[record.valid_from.astimezone(zone).date()].append(record)
    by_time = defaultdict(list)
    for day, entries in by_day.items():
        if day == latest.astimezone(zone).date() and (
            stock.operational.date_precision == "day"
            or any(r.operational.date_precision == "day" for r in entries)
        ):
            evidence.update(r.evidence_id for r in entries)
            flags.add("same_day_stock_movement_order_unknown")
            continue
        if len({r.operational.movement for r in entries}) > 1 and any(
            r.operational.date_precision == "day" for r in entries
        ):
            flags.add("coarse_date_consumption_precedes_receipts")
        for record in entries:
            when = record.valid_from
            if (
                record.operational.date_precision == "day"
                and record.operational.movement == "inflow"
            ):
                when = local_midnight(day + timedelta(days=1), zone) - timedelta(microseconds=1)
            by_time[when].append(record)
    at = None
    shortfall = None
    balance = Decimal(str(stock.quantity))
    for when, entries in sorted(by_time.items()):
        entries.sort(key=lambda r: (r.operational.movement != "outflow", r.evidence_id))
        if len(entries) > 1 and (len({r.operational.movement for r in entries}) > 1):
            flags.add("same_time_consumption_precedes_receipts")
        for record in entries:
            operation = record.operational
            evidence.add(record.evidence_id)
            amount = _amount(record, unit)
            if amount is None:
                flags.add("incompatible_unit")
                continue
            if operation.movement == "inflow":
                if operation.status not in {"confirmed", "received"}:
                    flags.add("unconfirmed_replenishment")
                    continue
                if (
                    operation.status != "received"
                    and operation.lead_time_days is not None
                    and (
                        operation.ordered_at is None
                        or operation.ordered_at + timedelta(days=operation.lead_time_days)
                        > record.valid_from
                    )
                ):
                    flags.add("receipt_lead_time_unverified")
                    continue
                balance += amount
            else:
                balance -= amount
                if balance < 0 and at is None:
                    at, shortfall = when, -balance
        if when < start and balance >= 0:
            at = shortfall = None
    return shortfall, at, unit, flags, evidence


def _dependency_flags(bom: list[EvidenceRevision], item: str, site: str) -> set[str]:
    graph = defaultdict(set)
    for revision in bom:
        record = revision.operational
        if record.site_id == site:
            graph[record.item_id].add(record.material_id)
    first_level = graph[item]
    if not any(graph.get(child) for child in first_level):
        return set()
    queue = [(item, frozenset())]
    visited = set()
    flags = {"unresolved_multilevel_bom"}
    while queue:
        node, path = queue.pop()
        if node in path:
            flags.add("bom_cycle")
            continue
        if len(path) >= 12 or len(visited) >= 200:
            flags.add("bom_traversal_limit")
            break
        key = (node, path)
        if key in visited:
            continue
        visited.add(key)
        queue.extend((child, path | {node}) for child in sorted(graph.get(node, ()), reverse=True))
    return flags


def _receipt_constraints(
    material_records: list[EvidenceRevision],
    revisions: tuple[EvidenceRevision, ...],
) -> list[EvidenceRevision]:
    result = []
    for item in material_records:
        record = item.operational
        if record.movement == "inflow" and record.supplier_id and record.lead_time_days is None:
            leads = {
                vendor.operational.lead_time_days
                for vendor in revisions
                if vendor.operational.dataset == "vendor_link"
                and vendor.operational.material_id == record.material_id
                and vendor.operational.site_id == record.site_id
                and vendor.operational.supplier_id == record.supplier_id
                and vendor.operational.lead_time_days is not None
                and _in_period(vendor, item.valid_from, item.valid_from + timedelta(seconds=1))
            }
            if leads:
                record = record.model_copy(update={"lead_time_days": max(leads)})
                item = item.model_copy(update={"operational": record})
        result.append(item)
    return result


def _allocation(
    allocations: list[EvidenceRevision],
    rows: list[ForecastRow],
    unit: str,
    total: Decimal,
) -> dict[str, Decimal]:
    amounts = {}
    for allocation in allocations:
        matches = [row for row in rows if _matches_item(allocation, row)]
        amount = _amount(allocation, unit)
        if len(matches) != 1 or amount is None or matches[0].series_id in amounts:
            return {}
        amounts[matches[0].series_id] = amount
    if len(amounts) == len(rows) and sum(amounts.values()) == total:
        return amounts
    return {}


def supply_groups(
    snapshot: ForecastSnapshot,
    revisions: tuple[EvidenceRevision, ...],
    *,
    complete_inputs: bool,
    locations: dict,
) -> tuple[ExposureGroup, ...]:
    bom = [revision for revision in revisions if revision.operational.dataset == "bom"]
    by_site = defaultdict(list)
    groups = []
    reservations = defaultdict(set)
    for revision in revisions:
        op = revision.operational
        if op.material_id and op.site_id:
            by_site[op.material_id, op.site_id].append(revision)
        if op.dataset == "alternate" and op.reservation_id:
            reservations[op.reservation_id].add(revision.record_id)
    by_month = defaultdict(list)
    zone = ZoneInfo(snapshot.timezone)
    for row in snapshot.rows:
        by_month[row.period].append(row)
    for period, rows in by_month.items():
        start, end = rows[0].target_start, rows[0].target_end
        effective_bom = [revision for revision in bom if _in_period(revision, start, end)]
        for (material, site), records in sorted(by_site.items()):
            by_series = {row.series_id: _dependency_links(records, row) for row in rows}
            eligible = [row for row in rows if by_series[row.series_id]]
            if not eligible:
                continue
            firm_alternates = [
                item
                for item in records
                if item.operational.dataset == "alternate"
                and item.operational.status == "confirmed"
                and item.quantity is not None
                and len(reservations.get(item.operational.reservation_id, ())) == 1
            ]
            reserved_deliveries = [
                item.model_copy(
                    update={
                        "operational": item.operational.model_copy(update={"movement": "inflow"}),
                    }
                )
                for item in firm_alternates
            ]
            adjusted_records = _receipt_constraints([*records, *reserved_deliveries], revisions)
            total, when, unit, flags, ids = _balance(adjusted_records, start, end, zone)
            membership_at = max(start, when) if when else start
            membership_end = membership_at + timedelta(microseconds=1)
            members = []
            for row in eligible:
                row_locations = locations[row.series_id, row.period]
                sites = {location for location, role in row_locations if role == "site"}
                if sites and site not in sites:
                    continue
                links = [
                    link
                    for link in by_series[row.series_id]
                    if _in_period(link, membership_at, membership_end)
                    and (
                        link.operational.dataset == "vendor_link"
                        or row.unit is None
                        or link.operational.finished_unit == row.unit
                    )
                ]
                if not links:
                    flags.add("bom_timing_or_finished_unit_mismatch")
                    continue
                if len(links) != 1 and any(link.operational.dataset == "bom" for link in links):
                    flags.add("ambiguous_bom")
                    continue
                ids.update(link.evidence_id for link in links)
                flags.update(_dependency_flags(effective_bom, row.item_id, site))
                members.append(row)
            if not members and not flags:
                continue
            if not complete_inputs:
                flags.add("incomplete_supply_inputs")
            alternatives = [
                r
                for r in records
                if r.operational.dataset == "alternate"
                and r.operational.status in {"approved", "confirmed"}
                and _in_period(r, membership_at, membership_end)
            ]
            if total is not None:
                for substitute in alternatives:
                    ids.add(substitute.evidence_id)
                    if substitute not in firm_alternates:
                        flags.add("substitute_availability_unknown")
                        if substitute.operational.reservation_id:
                            flags.add("ambiguous_substitutes")
            if not total and not flags:
                continue
            allocations = [
                r
                for r in records
                if r.operational.dataset == "allocation"
                and _in_period(r, membership_at, membership_end)
            ]
            uncertain = flags & {
                "incomplete_supply_inputs",
                "incompatible_unit",
                "unverified_warehouse_transfers",
                "no_opening_stock_snapshot",
                "same_day_stock_movement_order_unknown",
                "substitute_availability_unknown",
                "ambiguous_substitutes",
                "unresolved_multilevel_bom",
                "ambiguous_bom",
            }
            split = (
                _allocation(allocations, members, unit, total) if total and not uncertain else {}
            )
            ids.update(r.evidence_id for r in allocations)
            ids.update(r.evidence_id for r in records if r.operational.dataset == "vendor_link")
            if total and not split:
                flags.add("unallocated_shared_shortfall")
            qualified = {m.series_id for m in members if split.get(m.series_id) != 0}
            exceptions = [
                r
                for r in rows
                if r.series_id not in qualified
                and (
                    r in eligible
                    or any(
                        link.operational.material_id == material and _matches_item(link, r)
                        for link in effective_bom
                    )
                )
            ]
            groups.append(
                ExposureGroup(
                    group_id="group_"
                    + content_digest([snapshot.client_id, "supply", material, site, unit, period]),
                    kind="supply",
                    dependency=EntityRef(type="material", id=material),
                    period=period,
                    location_id=site,
                    location_role="site",
                    evidence_ids=tuple(sorted(ids)),
                    fact_version_ids=tuple(
                        sorted(
                            {
                                version
                                for row in members
                                for version in locations[row.series_id, row.period].get(
                                    (site, "site"), ()
                                )
                            }
                        )
                    ),
                    members=tuple(
                        member(r, float(split[r.series_id]) if split else None)
                        for r in members
                        if r.series_id in qualified
                    ),
                    exceptions=tuple(member(r) for r in exceptions),
                    flags=tuple(sorted(flags)),
                    event_at=when,
                    shared_shortfall=float(total)
                    if total is not None and members and not uncertain
                    else None,
                    unit=unit,
                )
            )
    return tuple(groups)
