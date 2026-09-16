"""Deliver usable evidence independently of best-effort artifact persistence."""
from . import harness, trace as trace_module


def deliver(envelope, trace, *, slug, root=None, no_save=False) -> dict:
    payload = envelope.model_dump(mode="json")
    persistence = {"status": "skipped" if no_save else "saved"}
    if not no_save:
        errors = []
        for key, saver, artifact in (
            ("answer_path", harness.save_answer, envelope),
            ("trace_path", trace_module.save_trace, trace),
        ):
            try:
                persistence[key] = str(saver(artifact, slug=slug, root=root))
            except (OSError, ValueError) as exc:
                errors.append(f"{key}: {type(exc).__name__}: {exc}"[:300])
        if errors:
            persistence.update(status="not_saved", reason="; ".join(errors)[:600])
    payload["persistence"] = persistence
    return payload
