"""Offline CASSIA Cell Ontology agreement scoring.

Natural-language labels are resolved from the pre-built mapping under
``evaluator/embedding_cache``. Missing labels can be extended through the
configured embedding endpoint, while scoring itself remains deterministic.
Resolved terms are scored with branch-weighted ontology distance::

    1 / (1 + d1 + d2 + branch_penalty * min(d1, d2))
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Optional
import urllib.error
import urllib.request

if TYPE_CHECKING:
    from evaluator.solver.CLSolver import CLSolver

Score = Optional[float]
MAPPING_PATH = Path(__file__).resolve().parents[1] / "embedding_cache" / "label_to_clid.tsv"
_CL_ID = re.compile(r"^CL:\d+$")
_TERM_RE = re.compile(r"^\[Term\]$")
_DEFAULT_EMBEDDING_BASE_URL = "https://api.openlux.ai/v1"
_EMBEDDING_BATCH_SIZE = 10
_EMBEDDING_TIMEOUT_SECONDS = 180
_EMBEDDING_MAX_RETRIES = 3


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _valid_cl_id(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _CL_ID.fullmatch(value) else None


def _label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _load_ontology_terms(path: Path) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if _TERM_RE.match(line):
            current = {}
        elif current is not None and line.startswith("id: "):
            current["id"] = line[4:].strip()
        elif current is not None and line.startswith("name: "):
            current["name"] = line[6:].strip()
        elif current is not None and line.startswith("synonym: "):
            match = re.search(r'"(.*?)"', line[9:])
            if match:
                current.setdefault("synonyms", []).append(match.group(1))
        elif current is not None and line.startswith("is_a: "):
            current.setdefault("parents", []).append(line[6:].split(" !", 1)[0].strip())
        elif not line and current and str(current.get("id", "")).startswith("CL:"):
            terms.append(current)
            current = None
    return terms


def _candidate_labels(term: dict[str, object]) -> list[str]:
    return list(dict.fromkeys([str(term.get("name", ""))] + [str(x) for x in term.get("synonyms", [])]))


def _load_json_cache(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _request_embeddings(endpoint: str, key: str, model: str, batch: list[str]) -> dict[str, object]:
    body = json.dumps({"model": model, "input": batch, "encoding_format": "float"}).encode()
    url = endpoint.rstrip("/") + "/embeddings"
    request = urllib.request.Request(
        url, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    )
    for attempt in range(_EMBEDDING_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=_EMBEDDING_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError("embedding response lacks a data list")
            return payload
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt >= _EMBEDDING_MAX_RETRIES:
                raise RuntimeError(f"embedding request failed with HTTP {error.code}") from error
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            if attempt >= _EMBEDDING_MAX_RETRIES:
                raise RuntimeError("embedding request failed after retries") from error
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("embedding request failed without a response")


def _embedding_vectors(texts: list[str], cache_path: Path, model: str, endpoint: str, key: str) -> list[list[float]]:
    cache = _load_json_cache(cache_path)
    missing = list(dict.fromkeys(text for text in texts if text not in cache))
    for start in range(0, len(missing), _EMBEDDING_BATCH_SIZE):
        batch = missing[start:start + _EMBEDDING_BATCH_SIZE]
        payload = _request_embeddings(endpoint, key, model, batch)
        items = sorted(payload["data"], key=lambda item: item["index"])
        if len(items) != len(batch) or any("embedding" not in item for item in items):
            raise ValueError("embedding response does not align with request")
        for text, item in zip(batch, items):
            cache[text] = item["embedding"]
        _atomic_write_json(cache_path, cache)
    return [cache[text] for text in texts]


def _cosine(first: list[float], second: list[float]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm) if first_norm and second_norm else 0.0


def _stable_choice(indices: list[int], label: str) -> int:
    digest = hashlib.sha256(_label_key(label).encode("utf-8")).digest()
    return indices[int.from_bytes(digest[:8], "big") % len(indices)]


def _term_depths(terms: list[dict[str, object]]) -> list[int]:
    ids = {str(term["id"]) for term in terms}
    parents = {str(term["id"]): [parent for parent in term.get("parents", []) if parent in ids] for term in terms}
    memo: dict[str, int] = {}

    def depth(term_id: str, visiting: set[str]) -> int:
        if term_id in memo:
            return memo[term_id]
        if term_id in visiting:
            return 10**9
        valid = parents.get(term_id, [])
        value = 0 if not valid else 1 + min(depth(parent, visiting | {term_id}) for parent in valid)
        memo[term_id] = value
        return value

    return [depth(str(term["id"]), set()) for term in terms]


def _map_labels(names: list[str], terms: list[dict[str, object]], cache_path: Path, model: str,
                endpoint: str, key: str, threshold: float = 0.55, margin: float = 0.0) -> dict[str, str]:
    labels = [label for term in terms for label in _candidate_labels(term)]
    term_indices = [index for index, term in enumerate(terms) for _ in _candidate_labels(term)]
    vectors = _embedding_vectors(labels + names, cache_path, model, endpoint, key)
    candidate_vectors, query_vectors = vectors[:len(labels)], vectors[len(labels):]
    depths = _term_depths(terms)
    result: dict[str, str] = {}
    for name, query in zip(names, query_vectors):
        scores = [0.0] * len(terms)
        for vector, term_index in zip(candidate_vectors, term_indices):
            scores[term_index] = max(scores[term_index], _cosine(query, vector))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        top = scores[order[0]]
        tied = [index for index, score in enumerate(scores) if abs(score - top) <= 1e-12]
        if len(tied) > 1:
            shallowest = min(depths[index] for index in tied)
            tied = [index for index in tied if depths[index] == shallowest]
        selected = _stable_choice(tied, name)
        second = max((score for index, score in enumerate(scores) if index not in tied), default=top)
        if top >= threshold and top - second >= margin:
            result[name] = str(terms[selected]["id"])
    return result


@lru_cache(maxsize=8)
def _load_mapping(path: str = str(MAPPING_PATH)) -> dict[str, str]:
    mapping: dict[str, str] = {}
    source = Path(path)
    if not source.is_file():
        return mapping
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = row.get("label_normalized") or row.get("label") or ""
            cl_id = _valid_cl_id(row.get("cl_id"))
            if key and cl_id:
                mapping[_mapping_key(key)] = cl_id
    return mapping


def label_to_cl_id(label: object, *, mapping_path: str | Path = MAPPING_PATH) -> Optional[str]:
    """Resolve a label from the frozen local map; never access the network."""
    if label is None or not isinstance(label, str) or not label.strip():
        return None
    value = label.strip()
    direct = _valid_cl_id(value)
    if direct:
        return direct
    return _load_mapping(str(mapping_path)).get(_mapping_key(value))


def _resolve_id(value: object, *, mapping_path: str | Path = MAPPING_PATH) -> object:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return ""
    return label_to_cl_id(value, mapping_path=mapping_path)


class CassiaScorer:
    """Score CL-ID pairs using branch-weighted Cell Ontology distance."""

    def __init__(self, solver: CLSolver | None = None, *, mapping_path: str | Path | None = MAPPING_PATH,
                 auto_extend_mapping: bool = True, ontology_path: str | Path | None = None,
                 embedding_model: str = "text-embedding-3-small", api_key_env: str = "OPENAI_API_KEY_LLM",
                 embedding_base_url: str | None = None, branch_penalty: float = 3.0) -> None:
        if isinstance(branch_penalty, bool) or not isinstance(branch_penalty, (int, float)):
            raise TypeError("branch_penalty must be a finite non-negative number")
        if not math.isfinite(branch_penalty) or branch_penalty < 0:
            raise ValueError("branch_penalty must be a finite non-negative number")
        if solver is None:
            from evaluator.solver.CLSolver import CLSolver
            solver = CLSolver()
        self.solver = solver
        self.mapping_path = Path(mapping_path) if mapping_path is not None else MAPPING_PATH
        self.auto_extend_mapping = auto_extend_mapping
        self.ontology_path = Path(ontology_path) if ontology_path else Path(__file__).resolve().parents[1] / "ref_data" / "cl.obo"
        self.embedding_model = embedding_model
        self.api_key_env = api_key_env
        self.embedding_base_url = (embedding_base_url or os.environ.get("OPENAI_BASE_URL")
                                   or _DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
        self.branch_penalty = float(branch_penalty)
        self.mapping_extension: dict[str, object] = {"enabled": auto_extend_mapping, "added": []}

    def ensure_mapping(self, values: Sequence[object]) -> list[str]:
        """Resolve missing labels using the built-in embedding mapper."""
        missing = self.unresolved_labels(values)
        self.mapping_extension["missing_before"] = list(missing)
        if not missing or not self.auto_extend_mapping:
            return missing
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"CASSIA auto-mapping requires environment variable {self.api_key_env}")
        if not self.ontology_path.is_file():
            raise FileNotFoundError(f"CASSIA ontology file not found: {self.ontology_path}")
        terms = _load_ontology_terms(self.ontology_path)
        if not terms:
            raise ValueError(f"CASSIA ontology contains no Cell Ontology terms: {self.ontology_path}")
        cache_path = self.mapping_path.parent / "embeddings.json"
        mapped = _map_labels(missing, terms, cache_path, self.embedding_model, self.embedding_base_url, key)
        if mapped:
            self._write_mapping_extension(mapped)
        _load_mapping.cache_clear()
        remaining = self.unresolved_labels(values)
        self.mapping_extension["added"] = [label for label in missing if label not in remaining]
        self.mapping_extension["remaining"] = remaining
        return remaining

    def _write_mapping_extension(self, mapped: dict[str, str]) -> None:
        """Merge valid generated mappings without replacing the existing cache."""
        rows: dict[str, dict[str, str]] = {}
        if self.mapping_path.is_file():
            with self.mapping_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    key = row.get("label_normalized", "")
                    if key and _valid_cl_id(row.get("cl_id")):
                        rows[key] = row
        for label, cl_id in mapped.items():
            rows[_mapping_key(label)] = {
                "label_normalized": _mapping_key(label), "label_original": label,
                "cl_id": cl_id, "score": "", "margin": "",
                "model": self.embedding_model, "policy_version": "embedding-v2-margin0-stable-tie",
            }
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.mapping_path.with_suffix(self.mapping_path.suffix + ".tmp")
        fields = ("label_normalized", "label_original", "cl_id", "score", "margin", "model", "policy_version")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])
        temporary.replace(self.mapping_path)

    @staticmethod
    def _cl_id(value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return ""
        return value.strip()

    def resolve_id(self, value: object) -> Optional[str]:
        return self._cl_id(_resolve_id(value, mapping_path=self.mapping_path))

    def unresolved_labels(self, values: Sequence[object]) -> list[str]:
        result = []
        for value in values:
            if isinstance(value, str) and value.strip() and self.resolve_id(value) is None:
                result.append(value.strip())
        return sorted(set(result), key=str.casefold)

    def score_one(self, first_id: object, second_id: object) -> Score:
        """Return the branch-weighted similarity of two ontology terms."""
        first = self.resolve_id(first_id)
        second = self.resolve_id(second_id)
        if first is None or second is None:
            return None
        if not first or not second:
            return 0.0
        graph = self.solver.cl_graph
        if first not in graph or second not in graph:
            return None

        candidates = self.solver.get_common_ancestor_candidates(first, second)
        if not candidates:
            return 0.0
        best = min(
            candidates,
            key=lambda item: (
                item["distance_first"]
                + item["distance_second"]
                + self.branch_penalty
                * min(item["distance_first"], item["distance_second"]),
                item["distance_first"] + item["distance_second"],
                item["cl_id"],
            ),
        )
        weighted_distance = (
            best["distance_first"]
            + best["distance_second"]
            + self.branch_penalty
            * min(best["distance_first"], best["distance_second"])
        )
        return 1.0 / (1.0 + weighted_distance)

    def score(self, first_ids: Sequence[object], second_ids: Sequence[object]) -> list[Score]:
        if len(first_ids) != len(second_ids):
            raise ValueError("first_ids and second_ids must have the same length")
        return [self.score_one(a, b) for a, b in zip(first_ids, second_ids)]

    def score_by_dataset(self, first_ids: Sequence[object], second_ids: Sequence[object], datasets: Sequence[str]) -> dict[str, Optional[float]]:
        if len(first_ids) != len(second_ids) or len(first_ids) != len(datasets):
            raise ValueError("first_ids, second_ids, and datasets must have the same length")
        grouped: dict[str, list[float]] = {}
        for first, second, dataset in zip(first_ids, second_ids, datasets):
            score = self.score_one(first, second)
            if score is None:
                return {dataset: None for dataset in sorted(set(datasets))}
            grouped.setdefault(dataset, []).append(score)
        return {dataset: sum(values) / len(values) for dataset, values in grouped.items()}


@lru_cache(maxsize=1)
def default_scorer() -> CassiaScorer:
    return CassiaScorer()


def evaluate_one(first_id: object, second_id: object) -> Score:
    return default_scorer().score_one(first_id, second_id)


def evaluate(first_ids: Sequence[object], second_ids: Sequence[object]) -> list[Score]:
    return [evaluate_one(a, b) for a, b in zip(first_ids, second_ids)]
