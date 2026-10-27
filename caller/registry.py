"""Canonical caller registry shared by dispatch and repository runner."""

from __future__ import annotations

from typing import Any


CALLER_REGISTRY: dict[str, dict[str, Any]] = {
    "anndict": {
        "module": "caller.anndict.anndict",
        "environment": "adt",
        "requires_species": False,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": True,
    },
    "cassia": {
        "module": "caller.cassia.cassia",
        "environment": "cassia",
        "requires_species": True,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": False,
    },
    "cellclick_v1": {
        "module": "caller.cellclick.cellclick",
        "environment": "cellclick",
        "protocol_version": "v1",
        "prediction_method": "cellclick",
        "output_namespace": "cellclick_v1",
        "requires_species": True,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": True,
    },
    "cellclick_v2": {
        "module": "caller.cellclick.cellclick",
        "environment": "cellclick",
        "protocol_version": "v2",
        "prediction_method": "cellclick",
        "output_namespace": "cellclick_v2",
        "requires_species": True,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": True,
    },
    "cellclick_v3": {
        "module": "caller.cellclick.cellclick", "environment": "cellclick",
        "protocol_version": "v3", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v3", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "cellclick_v4": {
        "module": "caller.cellclick_v4.cellclick", "environment": "cellclick",
        "protocol_version": "v4.4", "prediction_method": "cellclick",
        "supported_protocol_versions": ("v4.4", "v4.5"),
        "output_namespace": "cellclick_v4", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "celltypeagent": {
        "module": "caller.celltypeagent.celltypeagent",
        "environment": "CellTypeAgent",
        "requires_species": False,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": False,
    },
    "gptcelltype": {
        "module": "caller.celltypegpt.gptcelltype",
        "environment": "gptcelltype",
        "requires_species": False,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": False,
    },
    "lict": {
        "module": "caller.lict.lict",
        "environment": "lict",
        "requires_species": False,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": True,
    },
    "cellclick_v5": {
        "module": "caller.cellclick_v5.cellclick", "environment": "cellclick",
        "protocol_version": "v5", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v5", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "cellclick_v6": {
        "module": "caller.cellclick_v6.cellclick", "environment": "cellclick",
        "protocol_version": "v6", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v6", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "cellclick_v7": {
        "module": "caller.cellclick_v7.cellclick", "environment": "cellclick",
        "protocol_version": "v7", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v7", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "cellclick_v7_2": {
        "module": "caller.cellclick_v7_2.cellclick", "environment": "cellclick",
        "protocol_version": "v7.2", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v7_2", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "cellclick_v7_3": {
        "module": "caller.cellclick_v7_3.cellclick", "environment": "cellclick",
        "protocol_version": "v7.3", "prediction_method": "cellclick",
        "output_namespace": "cellclick_v7_3", "requires_species": True,
        "supports_expression_adapter": True, "requires_protocol_manifest": True,
    },
    "singler": {
        "module": "caller.singler.singler",
        "environment": "singler",
        "requires_species": False,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": False,
    },
    "sctype": {
        "module": "caller.sctype.sctype",
        "environment": "sctype",
        "requires_species": True,
        "supports_expression_adapter": True,
        "requires_protocol_manifest": False,
    },
}

CALLER_ALIASES = {"cellclick": "cellclick_v1"}


def normalize_caller_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("caller name must be a non-empty string")
    normalized = name.strip()
    canonical = CALLER_ALIASES.get(normalized, normalized)
    if canonical not in CALLER_REGISTRY:
        supported = ", ".join(sorted((*CALLER_REGISTRY, *CALLER_ALIASES)))
        raise ValueError(f"Unsupported caller {normalized!r}; supported: {supported}")
    return canonical


def caller_spec(name: str) -> dict[str, Any]:
    return CALLER_REGISTRY[normalize_caller_name(name)]


AVAILABLE_CALLERS = frozenset((*CALLER_REGISTRY, *CALLER_ALIASES))
