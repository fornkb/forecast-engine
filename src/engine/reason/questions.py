"""Typed onboarding questions and auditable operator confirmations."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from duckdb import DuckDBPyConnection

from engine.config.schema import ClientProfile
from engine.reason import store as records
from engine.reason.config import TargetConfig
from engine.reason.erp import require_reasoning
from engine.reason.facts import (
    OPERATOR_SOURCE,
    local_midnight,
    require_client,
    required_text,
    scalar_value,
)
from engine.reason.schemas import (
    CompanyFact,
    ContextError,
    EntityRef,
    FactValue,
    Question,
    QuestionAnswer,
    QuestionSpec,
    content_digest,
)
from engine.store.duckdb_store import EngineStore


def record_question(
    con: DuckDBPyConnection, client_id: str, spec: QuestionSpec, now: datetime
) -> Question:
    if (spec.answer_type == "object_id") != bool(spec.object_type):
        raise ContextError("Relationship questions require an object type")
    if any(month.day != 1 for month in spec.affected_months):
        raise ContextError("Affected months must be month-start dates")
    key = [client_id, spec.subject.model_dump(), spec.field, spec.location_role]
    question_id = "question_" + content_digest(key)
    existing = records.get_question(con, client_id, question_id)
    if existing is None:
        question = Question(
            client_id=client_id, question_id=question_id, spec=spec, status="pending", opened_at=now
        )
    else:
        previous = existing.spec
        if (spec.answer_type, spec.object_type) != (previous.answer_type, previous.object_type):
            raise ContextError(f"Question {question_id} has conflicting answer types")
        added_months = set(spec.affected_months) - set(previous.affected_months)
        added_facts = set(spec.candidate_fact_ids) - set(previous.candidate_fact_ids)
        scope = spec.model_copy(
            update={
                "affected_months": tuple(
                    sorted(set(previous.affected_months + spec.affected_months))
                ),
                "evidence_ids": tuple(sorted(set(previous.evidence_ids + spec.evidence_ids))),
                "event_ids": tuple(sorted(set(previous.event_ids + spec.event_ids))),
                "candidate_fact_ids": tuple(
                    sorted(set(previous.candidate_fact_ids + spec.candidate_fact_ids))
                ),
            }
        )
        question = existing.model_copy(
            update={
                "spec": scope,
                "status": "pending" if added_months or added_facts else existing.status,
            }
        )
    if question != existing:
        records.save_question(con, question)
    return question


def create_question(store: EngineStore, spec: QuestionSpec) -> Question:
    with records.write_transaction(store) as con:
        con.execute(records.REASON_DDL)
        return record_question(con, store.client_id, spec, datetime.now(UTC))


def list_questions(store: EngineStore, *, pending_only: bool = True) -> tuple[Question, ...]:
    if not store.path.exists():
        return ()
    with store.connect() as con:
        if not records.table_exists(con, "reason_questions"):
            return ()
        return records.read_questions(con, store.client_id, pending_only)


def get_question_answer(store: EngineStore, answer_id: str) -> QuestionAnswer:
    if store.path.exists():
        with store.connect() as con:
            if records.table_exists(con, "reason_question_answers"):
                answer = records.get_answer(con, store.client_id, answer_id)
                if answer:
                    return answer
    raise ContextError(f"Answer {answer_id!r} was not found for {store.client_id!r}")


def _parse_answer(spec: QuestionSpec, raw: str) -> FactValue:
    if spec.answer_type == "target":
        try:
            return TargetConfig.model_validate_json(raw)
        except ValueError as exc:
            raise ValueError(
                "target needs a JSON object with id, definition, and outcome_source"
            ) from exc
    return scalar_value(raw, spec.answer_type)


def confirm_question(
    profile: ClientProfile,
    store: EngineStore,
    question_id: str,
    value: str,
    actor: str,
    *,
    supersedes_answer_id: str | None = None,
) -> QuestionAnswer:
    require_client(profile, store)
    with records.write_transaction(store) as con:
        return confirm_in_transaction(
            con,
            profile,
            question_id,
            value,
            actor,
            supersedes_answer_id=supersedes_answer_id,
        )


def confirm_in_transaction(
    con: DuckDBPyConnection,
    profile: ClientProfile,
    question_id: str,
    value: str,
    actor: str,
    *,
    supersedes_answer_id: str | None = None,
) -> QuestionAnswer:
    config = require_reasoning(profile)
    question = None
    if records.table_exists(con, "reason_questions"):
        question = records.get_question(con, profile.client_id, question_id)
    if question is None:
        raise ContextError(f"Question {question_id!r} was not found for {profile.client_id!r}")
    spec = question.spec
    try:
        actor = required_text(actor, "actor")
        parsed = _parse_answer(spec, value)
    except ValueError as exc:
        raise ContextError(f"Invalid confirmation for {question_id}: {exc}") from exc
    previous = question.answer
    if (
        previous
        and question.status == "confirmed"
        and (previous.value, previous.actor) == (parsed, actor)
        and supersedes_answer_id in (None, previous.supersedes_answer_id)
    ):
        return previous
    expected = previous.answer_id if previous else None
    if supersedes_answer_id != expected:
        if previous:
            raise ContextError(
                f"Question already has an answer. To revise it, explicitly supersede {expected}."
            )
        raise ContextError("The question has no answer to supersede")

    now = datetime.now(UTC)
    revision = previous.revision + 1 if previous else 1
    digest = content_digest([question_id, revision, parsed, actor])
    answer_id = "answer_" + digest
    fact_id = "fact_" + content_digest([profile.client_id, OPERATOR_SOURCE, answer_id])
    zone = ZoneInfo(config.timezone)
    effective_from = (
        local_midnight(min(spec.affected_months), zone) if spec.affected_months else now
    )
    object_ref = None
    if spec.object_type:
        object_ref = EntityRef(type=spec.object_type, id=str(parsed))
    fact = CompanyFact(
        client_id=profile.client_id,
        fact_version_id=fact_id,
        source_id=OPERATOR_SOURCE,
        record_id=question_id,
        source_ref=f"question:{question_id}",
        source_version=str(revision),
        content_digest=digest,
        subject=spec.subject,
        relation=spec.field,
        object=object_ref,
        value=parsed if object_ref is None else None,
        location_role=spec.location_role,
        cardinality=spec.cardinality,
        effective_from=effective_from,
        known_at=now,
        captured_at=now,
        status="confirmed",
        actor=actor,
        resolves_fact_ids=spec.candidate_fact_ids,
    )
    answer = QuestionAnswer(
        answer_id=answer_id,
        question_id=question_id,
        revision=revision,
        value=parsed,
        actor=actor,
        confirmed_at=now,
        fact_version_id=fact_id,
        supersedes_answer_id=supersedes_answer_id,
        spec=spec,
    )
    records.save_fact(con, fact)
    records.save_answer(con, profile.client_id, answer)
    records.save_question(
        con, question.model_copy(update={"status": "confirmed", "answer": answer})
    )
    return answer
