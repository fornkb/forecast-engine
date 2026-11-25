"""Checksums for the finalized review and its pinned record contents."""

from engine.reason.schemas import ReviewManifest, ReviewSeal, content_digest


def manifest_digest(manifest: ReviewManifest) -> str:
    return content_digest(manifest.model_dump(mode="json", exclude={"seal"}))


def record_digests(forecast, batch, evidence, facts, checks, assessments) -> dict[str, str]:
    def digest(item):
        return content_digest(item.model_dump(mode="json"))

    return {
        f"snapshot:{forecast.snapshot_id}": digest(forecast),
        f"batch:{batch.batch_id}": digest(batch),
        **{f"evidence:{item.evidence_id}": digest(item) for item in evidence},
        **{f"fact:{item.fact_version_id}": digest(item) for item in facts},
        **{f"check:{item.check_id}": digest(item) for item in checks},
        **{
            f"assessment:{item.scenario_id}:{item.series_id}:{item.period}:{item.target}": digest(
                item
            )
            for item in assessments
        },
    }


def make_seal(manifest, forecast, batch, inputs, assessments) -> ReviewSeal:
    return ReviewSeal(
        manifest_digest=manifest_digest(manifest),
        records=record_digests(
            forecast,
            batch,
            inputs.evidence.values(),
            inputs.facts.values(),
            inputs.checks,
            assessments,
        ),
    )
