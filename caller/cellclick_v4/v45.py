"""Benchmark-side CellClick V4.5 compatibility layer.

The upstream V4 implementation remains frozen.  V4.5 adds the benchmark
protocol objects and routing metadata in this module, then delegates the
actual model/tool execution to the reviewed V4.4 implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CELLCLICK_SOURCE = REPOSITORY_ROOT / "src" / "cellclick"
if str(CELLCLICK_SOURCE) not in sys.path:
    sys.path.insert(0, str(CELLCLICK_SOURCE))

from cellclick.retry import retry_json
from cellclick.v4_5.models import TaskInput, ToolRequest
from cellclick.v4_5.tools import execute_tool


PROTOCOL_VERSION = "v4.5"
PROMPT_VERSION = "v4.5"
ANNOTATION_CANDIDATES = 5
MAX_QUERY_ROUNDS_TOTAL = 5
SUBTYPE_QUERY_ROUNDS_RESERVED = 2
MAX_REFINEMENT_ATTEMPTS = 1
MAX_IDENTITY_REOPENS = 1
REQUEST_RETRIES = 2
MAX_PARALLEL_CLUSTERS = 1


class Candidate(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    level: Literal["identity", "subtype"]
    parent_id: str | None = None
    relation_type: Literal["ontology_is_a", "operational_refinement"] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    proposal_status: Literal["provisional", "reviewed", "accepted", "invalid"] = "provisional"
    source: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def parent_required_for_subtype(self) -> "Candidate":
        if self.level == "subtype" and not self.parent_id:
            raise ValueError("subtype candidates require parent_id")
        if self.level == "subtype" and not self.relation_type:
            raise ValueError("subtype candidates require relation_type")
        return self


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_kind: Literal["marker_list", "expression", "reference", "tool_error"]
    availability: Literal["measured", "not_measured", "not_expressed", "error"]
    gene: str | None = None
    observation_id: str | None = None
    comparison: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class VerificationQuestion(BaseModel):
    question_id: str = Field(min_length=1)
    scope: Literal["identity", "subtype"]
    hypothesis_ids: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    status: Literal["open", "resolved", "unanswerable"] = "open"
    resolution_kind: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class VerificationReview(BaseModel):
    verification_scope: Literal["identity", "subtype"]
    verification_status: Literal["supported", "ambiguous", "contradicted", "insufficient"]
    supported_hypothesis_id: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    candidate_assessments: list[dict[str, Any]] = Field(default_factory=list)
    blocking_conflicts: list[str] = Field(default_factory=list)
    remaining_question_ids: list[str] = Field(default_factory=list)
    next_action: Literal["continue", "finish", "reopen_identity", "adjudicate"] = "adjudicate"
    reason: str = Field(min_length=1)


class FinalResult(BaseModel):
    execution_status: Literal["completed", "failed"]
    annotation_status: Literal["accepted", "unresolved"]
    verification_scope_completed: list[str] = Field(default_factory=list)
    selected_hypothesis_id: str | None = None
    accepted_level: Literal["identity", "subtype", "none"]
    identity_label: str | None = None
    subtype_label: str | None = None
    display_label: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_question_ids: list[str] = Field(default_factory=list)
    refinement_status: str = "not_attempted"
    refinement_stop_reason: str | None = None
    stop_reason: str = Field(min_length=1)


class SubtypeCandidate(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    relation_type: Literal["ontology_is_a", "operational_refinement"]
    rationale: str = Field(min_length=1)


class SubtypePlan(BaseModel):
    task_id: str = Field(min_length=1)
    parent_hypothesis_id: str = Field(min_length=1)
    candidates: list[SubtypeCandidate] = Field(min_length=1, max_length=ANNOTATION_CANDIDATES)
    query_genes: list[str] = Field(min_length=1, max_length=64)
    discriminating_question: str = Field(min_length=1)


class SubtypeReview(BaseModel):
    task_id: str = Field(min_length=1)
    verification_status: Literal["supported", "ambiguous", "insufficient"]
    selected_hypothesis_id: str | None = None
    selected_label: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class SubtypeReviewProtocolError(ValueError):
    """A syntactically valid review violated a v4.5 business constraint."""


@dataclass(frozen=True)
class IdentityResolution:
    """Deterministic link between the upstream decision and an identity candidate."""

    candidate: dict[str, Any] | None
    method: str
    skip_reason: str | None = None


def resolve_identity_candidate(identity_record: dict[str, Any]) -> IdentityResolution:
    """Resolve an upstream identity decision without relying on fuzzy label matching.

    The final display label may intentionally be broader than a candidate (for
    example, ``Lymphocyte`` rather than ``T cell``).  Such an adjudicated
    parent-level decision must not be silently refined back into a child label.
    """
    a2 = identity_record.get("a2") or {}
    candidates = [item for item in a2.get("candidates", []) if isinstance(item, dict)]
    if not candidates:
        return IdentityResolution(None, "unresolved", "identity_candidate_missing")

    by_id = {str(item.get("hypothesis_id")): item for item in candidates
             if str(item.get("hypothesis_id") or "").strip()}
    for key in ("accepted_hypothesis_id", "selected_hypothesis_id", "supported_hypothesis_id"):
        candidate_id = str(identity_record.get(key) or "").strip()
        if candidate_id in by_id:
            return IdentityResolution(by_id[candidate_id], f"explicit_{key}")

    prediction = str(identity_record.get("prediction") or "").strip()
    exact = [item for item in candidates if str(item.get("label") or "").strip() == prediction]
    if len(exact) == 1:
        return IdentityResolution(exact[0], "exact_label")
    if len(exact) > 1:
        return IdentityResolution(None, "unresolved", "identity_candidate_ambiguous")

    parent_matches = [item for item in candidates
                      if str(item.get("parent_label") or "").strip() == prediction]
    if len(parent_matches) == 1:
        return IdentityResolution(None, "parent_label", "parent_level_final_decision")
    if len(parent_matches) > 1:
        return IdentityResolution(None, "unresolved", "identity_candidate_ambiguous")
    if len(candidates) == 1:
        return IdentityResolution(candidates[0], "unique_candidate_fallback")
    return IdentityResolution(None, "unresolved", "identity_candidate_ambiguous")


@dataclass
class TaskState:
    task_id: str
    dataset: str
    tissue: str
    phase: str = "identity_candidates"
    candidate_registry: dict[str, Candidate] = field(default_factory=dict)
    evidence_by_id: dict[str, EvidenceRecord] = field(default_factory=dict)
    questions_by_id: dict[str, VerificationQuestion] = field(default_factory=dict)
    identity_review: VerificationReview | None = None
    subtype_review: VerificationReview | None = None
    registry_version: int = 0
    refinement_attempted: bool = False
    identity_reopen_count: int = 0
    query_rounds_used: int = 0
    query_rounds_remaining: int = MAX_QUERY_ROUNDS_TOTAL
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)

    def register_candidate(self, candidate: Candidate) -> None:
        if candidate.hypothesis_id in self.candidate_registry:
            raise ValueError(f"duplicate hypothesis_id: {candidate.hypothesis_id}")
        if candidate.parent_id and candidate.parent_id not in self.candidate_registry:
            raise ValueError(f"unknown parent_id: {candidate.parent_id}")
        if candidate.parent_id == candidate.hypothesis_id:
            raise ValueError("candidate cannot be its own parent")
        self.candidate_registry[candidate.hypothesis_id] = candidate
        self.registry_version += 1

    def register_evidence(self, evidence: EvidenceRecord) -> None:
        if evidence.task_id != self.task_id:
            raise ValueError("evidence belongs to another task")
        if evidence.evidence_id in self.evidence_by_id:
            raise ValueError(f"duplicate evidence_id: {evidence.evidence_id}")
        self.evidence_by_id[evidence.evidence_id] = evidence

    def checkpoint(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "dataset": self.dataset, "tissue": self.tissue,
            "phase": self.phase, "registry_version": self.registry_version,
            "candidate_registry": {key: value.model_dump() for key, value in self.candidate_registry.items()},
            "evidence_by_id": {key: value.model_dump() for key, value in self.evidence_by_id.items()},
            "questions_by_id": {key: value.model_dump() for key, value in self.questions_by_id.items()},
            "refinement_attempted": self.refinement_attempted,
            "identity_reopen_count": self.identity_reopen_count,
            "query_rounds_used": self.query_rounds_used,
            "query_rounds_remaining": self.query_rounds_remaining,
            "phase_history": self.phase_history, "protocol_errors": self.protocol_errors,
        }


def route_after_review(state: TaskState, review: VerificationReview) -> str:
    """Return the next program-owned phase for a validated review."""
    allowed = set(state.candidate_registry)
    if review.supported_hypothesis_id and review.supported_hypothesis_id not in allowed:
        raise ValueError("review references an unknown supported hypothesis")
    if review.verification_scope == "identity":
        state.identity_review = review
        if review.next_action == "reopen_identity" and state.identity_reopen_count < MAX_IDENTITY_REOPENS:
            state.identity_reopen_count += 1
            state.phase = "identity_candidates"
            return state.phase
        if review.verification_status == "supported" and review.supported_hypothesis_id:
            state.phase = "subtype_candidates"
            return state.phase
    else:
        state.subtype_review = review
    state.phase = "adjudication"
    return state.phase


def protocol_fingerprint() -> str:
    payload = {"protocol": PROTOCOL_VERSION, "prompt": PROMPT_VERSION,
               "candidates": ANNOTATION_CANDIDATES, "rounds": MAX_QUERY_ROUNDS_TOTAL,
               "subtype_reserved": SUBTYPE_QUERY_ROUNDS_RESERVED,
               "refinement_attempts": MAX_REFINEMENT_ATTEMPTS,
               "identity_reopens": MAX_IDENTITY_REOPENS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def contract_fingerprint() -> str:
    payload = {"candidate": Candidate.model_json_schema(), "evidence": EvidenceRecord.model_json_schema(),
               "question": VerificationQuestion.model_json_schema(),
               "review": VerificationReview.model_json_schema(), "result": FinalResult.model_json_schema(),
               "subtype_plan": SubtypePlan.model_json_schema(),
               "subtype_review": SubtypeReview.model_json_schema()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _subtype_phase(*, client: Any, task: TaskInput, identity_record: dict[str, Any],
                         model: str, temperature: float, retries: int,
                         expression_file: str | Path, expression_manifest: str | Path) -> dict[str, Any]:
    """Run the v4.5 child-candidate, query, and review phases."""
    prediction = str(identity_record.get("prediction") or "").strip()
    resolution = resolve_identity_candidate(identity_record)
    parent = resolution.candidate
    if parent is None:
        raise ValueError(f"v4.5 subtype phase requires a resolvable identity candidate: {resolution.skip_reason}")
    parent_id = str(parent.get("hypothesis_id"))
    base = {
        "task_id": task.task_id, "dataset": task.dataset, "species": task.species,
        "tissue": task.tissue, "ordered_markers": task.ordered_markers,
        "identity_prediction": prediction, "identity_candidate": parent,
    }

    raw_plan, _ = await retry_json(
        client, system=("You are the CellClick v4.5 subtype-candidate planner. Return JSON only. "
                        "Propose biologically meaningful child labels of the accepted identity. "
                        "Use only the supplied marker genes and do not use reference labels."),
        user=json.dumps({**base, "current_stage": "subtype_candidates",
                         "instruction": "Propose child candidates and a small discriminating marker query."},
                        ensure_ascii=False), schema=SubtypePlan.model_json_schema(), model=model,
        temperature=temperature, retries=retries, strict=True,
    )
    plan = SubtypePlan.model_validate(raw_plan)
    if plan.task_id != task.task_id or plan.parent_hypothesis_id != parent_id:
        raise ValueError("subtype plan does not match the accepted identity")
    normalized_genes = [gene.strip() for gene in plan.query_genes if gene.strip()]
    if len(normalized_genes) != len(set(gene.casefold() for gene in normalized_genes)):
        raise ValueError("subtype query contains duplicate genes")
    if not normalized_genes:
        raise ValueError("subtype query contains no genes")
    for candidate in plan.candidates:
        if candidate.parent_id != parent_id:
            raise ValueError("subtype candidate has an invalid parent")

    task_request = ToolRequest(
        request_id=f"subtype_{task.task_id}", task_id=task.task_id,
        hypothesis_ids=[candidate.hypothesis_id for candidate in plan.candidates],
        tool_name="get_marker_statistics", purpose="distinguish child cell-type candidates",
        discriminating_question=plan.discriminating_question,
        arguments={"genes": normalized_genes, "comparison": "target_vs_rest"},
    )
    observation = execute_tool(task, task_request, expression_file=expression_file,
                               expression_manifest=expression_manifest)
    if observation.status != "success":
        raise RuntimeError("v4.5 subtype query failed: " + ";".join(observation.limitations))
    subtype_evidence_ids = [f"EV_{observation.observation_id}"]
    candidate_by_id = {candidate.hypothesis_id: candidate for candidate in plan.candidates}

    def validate_review_payload(value: dict[str, Any]) -> None:
        """Validate review semantics inside retry_json so invalid answers are repaired."""
        try:
            parsed = SubtypeReview.model_validate(value)
            if parsed.task_id != task.task_id:
                raise ValueError("subtype review does not match task")
            if parsed.selected_hypothesis_id is not None and parsed.selected_hypothesis_id not in candidate_by_id:
                raise ValueError("subtype review selected an unknown candidate")
            if not set(parsed.supporting_evidence_ids).issubset(set(subtype_evidence_ids)):
                raise ValueError("subtype review cites unknown evidence")
            if parsed.verification_status == "supported":
                if (not parsed.selected_hypothesis_id
                        or parsed.selected_label != candidate_by_id[parsed.selected_hypothesis_id].label):
                    raise ValueError("supported subtype review must select a planned candidate")
        except Exception as error:
            raise SubtypeReviewProtocolError(str(error)) from error

    try:
        raw_review, _ = await retry_json(
            client,
            system=("You are the CellClick v4.5 subtype reviewer. Return JSON only. "
                    "Select a child candidate only when the expression statistics support it; "
                    "otherwise return insufficient. Never promote a candidate without evidence."),
            user=json.dumps({**base, "current_stage": "subtype_review", "subtype_plan": plan.model_dump(),
                             "tool_observation": observation.model_dump(),
                             "allowed_evidence_ids": subtype_evidence_ids}, ensure_ascii=False),
            schema=SubtypeReview.model_json_schema(), model=model, temperature=temperature,
            retries=retries, strict=True, validator=validate_review_payload,
        )
        review = SubtypeReview.model_validate(raw_review)
    except SubtypeReviewProtocolError as error:
        # A malformed evidence reference must never promote a child or fail the
        # whole task. Preserve the accepted identity and leave a clear audit trail.
        review = SubtypeReview(
            task_id=task.task_id, verification_status="insufficient",
            supporting_evidence_ids=[], rationale="Subtype review rejected after validation retries: " + str(error),
        )
        return {"plan": plan.model_dump(), "request": task_request.model_dump(),
                "observation": observation.model_dump(), "review": review.model_dump(),
                "evidence_ids": subtype_evidence_ids, "protocol_error": str(error)}
    if not review.supporting_evidence_ids and review.verification_status == "supported":
        review.supporting_evidence_ids = subtype_evidence_ids
    return {"plan": plan.model_dump(), "request": task_request.model_dump(),
            "observation": observation.model_dump(), "review": review.model_dump(),
            "evidence_ids": subtype_evidence_ids, "protocol_error": None}


def annotate_cluster(*, cluster_id: str, dataset: str, species: str, tissue: str, markers: list[str],
                     client: Any, model: str, temperature: float, retries: int,
                     work_dir: Path | None = None, expression_file: str | Path | None = None,
                     expression_manifest: str | Path | None = None,
                     prompt_version: str = PROMPT_VERSION,
                     protocol_version: str = PROTOCOL_VERSION,
                     subtype_client_factory: Callable[[], Any] | None = None,
                     **kwargs: Any) -> dict[str, Any]:
    """Run identity and subtype stages with clients bound to their own event loops."""
    if prompt_version != PROMPT_VERSION or protocol_version != PROTOCOL_VERSION:
        raise ValueError("CellClick V4.5 requires protocol_version=v4.5 and prompt_version=v4.5")
    if not expression_file or not expression_manifest:
        raise ValueError("CellClick V4.5 requires expression_file and expression_manifest")
    from cellclick.v4_5.graph import annotate_cluster as upstream_annotate

    # The upstream identity workflow owns and closes ``client`` in the same
    # event loop that performs its requests.  Do not retain it for subtype work.
    result = upstream_annotate(
        cluster_id=cluster_id, dataset=dataset, species=species, tissue=tissue, markers=markers,
        client=client, model=model, temperature=temperature, retries=retries, work_dir=work_dir,
        prompt_version="v4.5", protocol_version="v4.5", expression_file=expression_file,
        expression_manifest=expression_manifest, **kwargs,
    )
    record = result.get("final_record") or {}
    status = result.get("status")
    task = TaskInput(task_id=cluster_id, cluster_id=cluster_id, dataset=dataset, species=species,
                     tissue=tissue, ordered_markers=markers)
    subtype = None
    identity_resolution = IdentityResolution(None, "not_applicable")
    state = TaskState(task_id=cluster_id, dataset=dataset, tissue=tissue,
                      phase="identity_review" if status else "failed")
    if status == "validated":
        identity_resolution = resolve_identity_candidate(record)
        record["v45_identity_resolution"] = {
            "method": identity_resolution.method,
            "candidate_hypothesis_id": (identity_resolution.candidate or {}).get("hypothesis_id"),
            "skip_reason": identity_resolution.skip_reason,
        }

        identity_candidate = identity_resolution.candidate
        if identity_candidate is not None:
            state.register_candidate(Candidate(
                hypothesis_id=str(identity_candidate["hypothesis_id"]),
                label=str(identity_candidate["label"]), level="identity", source="v4.4_identity",
            ))
        if identity_candidate is None:
            state.phase = "adjudication"
            state.phase_history.extend([
                {"phase": "identity_review", "status": status},
                {"phase": "subtype", "status": "skipped",
                 "reason": identity_resolution.skip_reason},
            ])
            record.setdefault("decision_history", []).append({
                "stage": "subtype", "status": "skipped",
                "reason": identity_resolution.skip_reason,
                "identity_resolution": identity_resolution.method,
            })
        else:
            if subtype_client_factory is None:
                subtype_client_factory = type(client)

            async def run_subtype() -> dict[str, Any]:
                from cellclick.llm import close_json_client

                subtype_client = subtype_client_factory()
                try:
                    return await _subtype_phase(
                        client=subtype_client, task=task, identity_record=record, model=model,
                        temperature=temperature, retries=retries, expression_file=expression_file,
                        expression_manifest=expression_manifest,
                    )
                finally:
                    await close_json_client(subtype_client)

            state.refinement_attempted = True
            state.phase = "subtype_candidates"
            subtype = asyncio.run(run_subtype())
            state.query_rounds_used = 1
            state.query_rounds_remaining = MAX_QUERY_ROUNDS_TOTAL - 1
            review = subtype["review"]
            for child in subtype["plan"]["candidates"]:
                state.register_candidate(Candidate(
                    hypothesis_id=child["hypothesis_id"], label=child["label"], level="subtype",
                    parent_id=child["parent_id"], relation_type=child["relation_type"],
                    source="v4.5_subtype_candidates", proposal_status="reviewed",
                ))
            state.phase = "adjudication"
            state.phase_history.extend([
                {"phase": "identity_review", "status": status},
                {"phase": "subtype_candidates", "status": "completed",
                 "candidate_count": len(subtype["plan"]["candidates"])},
                {"phase": "subtype_query", "status": subtype["observation"]["status"]},
                {"phase": "subtype_review", "status": review["verification_status"]},
            ])
            if subtype.get("protocol_error"):
                state.protocol_errors.append(subtype["protocol_error"])
                record["v45_protocol_errors"] = list(state.protocol_errors)
                record.setdefault("decision_history", []).append({
                    "stage": "subtype_review", "status": "rejected_protocol_error",
                    "error": subtype["protocol_error"],
                })
            if review["verification_status"] == "supported":
                record["prediction"] = review["selected_label"]
                result["status"] = "validated"
            record.setdefault("decision_history", []).extend([
                {"stage": "subtype_candidates", "status": "reviewed",
                 "labels": [item["label"] for item in subtype["plan"]["candidates"]]},
                {"stage": "subtype_query", "status": subtype["observation"]["status"],
                 "evidence_ids": subtype["evidence_ids"]},
                {"stage": "subtype_review", "status": review["verification_status"],
                 "label": review.get("selected_label"), "rationale": review["rationale"]},
            ])
            record["v45_call_counts"] = {"subtype_candidates": 1, "subtype_query": 1, "subtype_review": 1}
    a4 = record.get("a4") or {}
    parent_label = a4.get("parent_label") or record.get("prediction")
    final_review = (subtype or {}).get("review", {})
    accepted_subtype = final_review.get("verification_status") == "supported"
    final = FinalResult(
        execution_status="completed", annotation_status="accepted" if result.get("status") == "validated" else "unresolved",
        verification_scope_completed=["identity", "subtype"] if accepted_subtype else ["identity"],
        selected_hypothesis_id=final_review.get("selected_hypothesis_id") if accepted_subtype else None,
        accepted_level="subtype" if accepted_subtype else "identity",
        identity_label=parent_label, subtype_label=record.get("prediction") if accepted_subtype else None,
        display_label=record.get("prediction"),
        supporting_evidence_ids=(list(a4.get("observed_support_ids", [])) +
                                 (list((subtype or {}).get("evidence_ids", [])) if accepted_subtype else [])),
        refinement_status="accepted" if accepted_subtype else ("skipped" if status == "validated" and subtype is None else "insufficient"),
        refinement_stop_reason=(None if accepted_subtype else
                                (identity_resolution.skip_reason if status == "validated" and subtype is None
                                 else "subtype_review_insufficient")),
        stop_reason=("subtype_review_protocol_error" if (subtype or {}).get("protocol_error") else
                     "subtype_review_completed" if subtype else
                     (identity_resolution.skip_reason if status == "validated" and identity_resolution.skip_reason
                      else record.get("stop_reason", "completed"))),
    )
    record["v45_subtype"] = subtype
    record["v45_final_result"] = final.model_dump()
    record["protocol_version"] = PROTOCOL_VERSION
    record["protocol_fingerprint"] = protocol_fingerprint()
    record["contract_fingerprint"] = contract_fingerprint()
    result["final_record"] = record
    result["protocol_version"] = PROTOCOL_VERSION
    if work_dir is not None:
        trace_path = Path(work_dir) / "traces" / f"{cluster_id}.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["protocol_version"] = PROTOCOL_VERSION
            trace["prompt_version"] = PROMPT_VERSION
            trace["final_record"] = record
            trace["v45_subtype"] = subtype
            trace["workflow_version"] = "marker-statistics-v4.5-identity-subtype-adjudication"
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checkpoint_dir = Path(work_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / f"{cluster_id}.json").write_text(
            json.dumps({"protocol_version": PROTOCOL_VERSION,
                        "protocol_fingerprint": protocol_fingerprint(),
                        "contract_fingerprint": contract_fingerprint(),
                        "state": state.checkpoint(), "final_result": final.model_dump()},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
