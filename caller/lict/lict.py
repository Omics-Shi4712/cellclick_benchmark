"""LICT-style multi-model discussion and independent final evaluation.

The upstream LICT R package hard-codes five providers and a GPT model.  This
adapter keeps ``src/LICT`` untouched and implements the same high-level idea
over the benchmark's label-free marker TSV using an OpenAI-compatible API.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

OUTPUT_FIELDS = ("source_row_id", "dataset", "tissue", "prediction", "method", "model", "task_id")
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = list(csv.DictReader(handle, delimiter="\t"))
    required = {"source_row_id", "dataset", "tissue", "marker"}
    if not result or set(result[0]) != required:
        raise ValueError("LICT query must contain exactly source_row_id, dataset, tissue, marker")
    if len({row["source_row_id"] for row in result}) != len(result):
        raise ValueError("LICT query contains duplicate source_row_id values")
    if any(not row["marker"].strip() for row in result):
        raise ValueError("LICT query contains an empty marker list")
    return result


def _model_entries(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"LICT {field} must be a non-empty list")
    entries: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            entries.append({"name": item.strip(), "model": item.strip()})
        elif isinstance(item, Mapping):
            model = item.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"LICT {field} entries require a non-empty model")
            name = item.get("name", model)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"LICT {field} entry name must be non-empty")
            entry = {"name": name.strip(), "model": model.strip()}
            for key in ("temperature", "max_tokens"):
                if key in item:
                    entry[key] = item[key]
            entries.append(entry)
        else:
            raise ValueError(f"LICT {field} entries must be strings or mappings")
    return entries


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        value = getattr(message, "content", None) if message is not None else None
        if value is not None:
            return str(value)
    if isinstance(response, Mapping):
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            pass
    raise ValueError("OpenAI-compatible response has no message content")


def _json_value(text: str) -> Any:
    candidate = text.strip()
    match = _JSON_BLOCK.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = min((p for p in (candidate.find("{"), candidate.find("[")) if p >= 0), default=-1)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise ValueError("Model response is not valid JSON")


def _chat(model: Mapping[str, Any], messages: list[dict[str, str]], config: Mapping[str, Any]) -> str:
    try:
        import openai
    except ImportError as error:
        raise RuntimeError("LICT requires the openai Python package in its Conda environment") from error
    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"LICT API key environment variable {api_key_env!r} is not set")
    base_url = config.get("base_url")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["api_base"] = str(base_url).rstrip("/")
    # openai 0.28 (the pinned LICT environment) uses the module-level API.
    openai.api_key = kwargs["api_key"]
    if "api_base" in kwargs:
        openai.api_base = kwargs["api_base"]
    request: dict[str, Any] = {"model": model["model"], "messages": messages}
    for key in ("temperature", "max_tokens"):
        if key in model:
            request[key] = model[key]
    response = openai.ChatCompletion.create(**request)
    return _content(response)


def _discussion_prompt(rows: list[dict[str, str]], species: str, tissue: str) -> str:
    payload = [{"source_row_id": row["source_row_id"], "markers": [x.strip() for x in row["marker"].split(",") if x.strip()]} for row in rows]
    return (
        "You are one cell-type annotation expert. Identify one cell type for each row. "
        "Use only the supplied marker genes and return JSON object mapping each source_row_id "
        "to a concise cell-type label; do not add explanations. "
        f"Species: {species}. Tissue/context: {tissue}. Rows: {json.dumps(payload, ensure_ascii=False)}"
    )


def _evaluator_prompt(row: Mapping[str, str], candidates: list[Mapping[str, str]], species: str) -> str:
    return (
        "You are the final cell-type evaluator. Choose the best single label from the candidate "
        "answers, using the marker genes as evidence. Return JSON exactly as "
        '{"prediction":"..."}; no explanation. '
        f"Species: {species}. Tissue: {row['tissue']}. Markers: {row['marker']}. "
        f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
    )


def run_task(task: dict[str, Any]) -> None:
    config = task.get("method_config", {})
    if not isinstance(config, Mapping):
        raise ValueError("LICT method_config must be a mapping")
    rows = _rows(Path(task["query_tsv"]))
    discussion = _model_entries(config.get("discussion_models"), "discussion_models")
    evaluator_list = _model_entries([config.get("evaluator_model")], "evaluator_model")
    evaluator = evaluator_list[0]
    species = str(config.get("species", "human"))
    work_dir = Path(task["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict[str, str]]] = {row["source_row_id"]: [] for row in rows}
    prompt = _discussion_prompt(rows, species, task.get("tissue_context", rows[0]["tissue"]))
    traces: dict[str, Any] = {"discussion_models": [m["model"] for m in discussion], "evaluator_model": evaluator["model"], "candidates": {}}
    for index, model in enumerate(discussion):
        text = _chat(model, [{"role": "user", "content": prompt}], config)
        (work_dir / f"discussion_{index + 1:03d}.response.txt").write_text(text, encoding="utf-8")
        value = _json_value(text)
        if not isinstance(value, Mapping):
            raise ValueError(f"LICT discussion model {model['model']!r} did not return a JSON object")
        for row in rows:
            label = value.get(row["source_row_id"])
            if isinstance(label, str) and label.strip():
                candidates[row["source_row_id"]].append({"model": model["name"], "prediction": label.strip()})
    if any(not values for values in candidates.values()):
        missing = [key for key, values in candidates.items() if not values]
        raise ValueError(f"LICT has no discussion prediction for source_row_id(s): {missing}")
    predictions: dict[str, str] = {}
    for row in rows:
        text = _chat(evaluator, [{"role": "user", "content": _evaluator_prompt(row, candidates[row["source_row_id"]], species)}], config)
        value = _json_value(text)
        if not isinstance(value, Mapping) or not isinstance(value.get("prediction"), str) or not value["prediction"].strip():
            raise ValueError(f"LICT evaluator {evaluator['model']!r} returned no prediction for {row['source_row_id']}")
        predictions[row["source_row_id"]] = value["prediction"].strip()
        (work_dir / f"evaluator_{row['source_row_id']}.response.txt").write_text(text, encoding="utf-8")
    traces["candidates"] = candidates
    (work_dir / "lict_trace.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output = Path(task["output_tsv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({"source_row_id": row["source_row_id"], "dataset": row["dataset"], "tissue": row["tissue"], "prediction": predictions[row["source_row_id"]], "method": "lict", "model": evaluator["model"], "task_id": task["task_id"]})

