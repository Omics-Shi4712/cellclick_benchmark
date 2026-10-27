"""Strict JSON Schema compatibility for CellClick V5 provider requests."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


class SchemaCompatibilityError(ValueError):
    """Raised when a V5 response schema cannot be made provider-compatible."""


_FREE_FORM_MARKER = "__cellclick_free_form_any__"


def _resolve_local_ref(reference: str, definitions: Mapping[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not reference.startswith(prefix):
        raise SchemaCompatibilityError(f"Unsupported JSON Schema reference: {reference!r}")
    name = reference[len(prefix):]
    target = definitions.get(name)
    if not isinstance(target, dict):
        raise SchemaCompatibilityError(f"Unknown JSON Schema definition: {reference!r}")
    return copy.deepcopy(target)


def provider_strict_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Inline local definitions and validate the strict-schema object invariant.

    Some OpenAI-compatible providers reject schemas with ``$defs``/``$ref`` even
    when the same schema is accepted by the OpenAI API.  V5 uses strict response
    schemas, so retain that contract while sending a self-contained schema.
    """
    result = copy.deepcopy(dict(schema))
    definitions = result.pop("$defs", {})
    if not isinstance(definitions, dict):
        raise SchemaCompatibilityError("JSON Schema $defs must be an object")

    def transform(node: Any, resolving: tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [transform(value, resolving) for value in node]
        if not isinstance(node, dict):
            return node

        reference = node.pop("$ref", None)
        if reference is not None:
            if not isinstance(reference, str):
                raise SchemaCompatibilityError("JSON Schema $ref must be a string")
            name = reference.removeprefix("#/$defs/")
            if name in resolving:
                raise SchemaCompatibilityError(f"Recursive JSON Schema reference: {reference!r}")
            resolved = transform(_resolve_local_ref(reference, definitions), resolving + (name,))
            resolved.update(node)
            node = resolved

        for key in ("title", "default", "minLength", "maxLength", "minItems", "maxItems"):
            node.pop(key, None)
        if "const" in node:
            node["enum"] = [node.pop("const")]
        for key, value in list(node.items()):
            node[key] = transform(value, resolving)

        if isinstance(node.get("additionalProperties"), dict):
            raise SchemaCompatibilityError("Strict V5 response schema cannot contain dynamic object keys")
        # ``dict[str, Any]`` becomes an object with ``additionalProperties:
        # true``. Preserve that distinction before strict normalization erases
        # it so it can be rejected explicitly below.
        is_free_form_any = node.get("type") == "object" and node.get("additionalProperties") is True
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            if not isinstance(properties, dict):
                raise SchemaCompatibilityError("JSON Schema object properties must be an object")
            node["additionalProperties"] = False
            node["required"] = list(properties)
        if is_free_form_any:
            node[_FREE_FORM_MARKER] = True
        return node

    transformed = transform(result)

    def reject_free_form_fields(node: Any) -> None:
        """Reject arbitrary maps instead of silently deleting business fields."""
        if isinstance(node, list):
            for value in node:
                reject_free_form_fields(value)
        elif isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for name in list(properties):
                    value = properties[name]
                    is_free_form_object = isinstance(value, dict) and value.get(_FREE_FORM_MARKER)
                    is_free_form_object_array = (
                        isinstance(value, dict) and value.get("type") == "array"
                        and isinstance(value.get("items"), dict) and value["items"].get(_FREE_FORM_MARKER)
                    )
                    if is_free_form_object or is_free_form_object_array:
                        raise SchemaCompatibilityError(
                            f"Strict V5 response schema contains unsupported free-form field: {name}"
                        )
                    else:
                        reject_free_form_fields(value)
            for value in node.values():
                reject_free_form_fields(value)

            node.pop(_FREE_FORM_MARKER, None)

    reject_free_form_fields(transformed)

    def validate(node: Any) -> None:
        if isinstance(node, list):
            for value in node:
                validate(value)
        elif isinstance(node, dict):
            if "$ref" in node or "$defs" in node:
                raise SchemaCompatibilityError("Provider schema must not contain $ref or $defs")
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties", {})
                if node.get("additionalProperties") is not False or set(node.get("required", ())) != set(properties):
                    raise SchemaCompatibilityError("Strict object schema requires every property exactly once")
            for value in node.values():
                validate(value)

    validate(transformed)
    return transformed


class V5SchemaCompatibleClient:
    """Delegate client that normalizes only V5 strict response schemas."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    @property
    def last_usage(self):
        return getattr(self._delegate, "last_usage", {})

    async def complete_json(self, *, system: str, user: str, schema: dict[str, Any], model: str,
                            temperature: float, strict: bool = False) -> tuple[dict[str, Any], str]:
        # V5 responses use fixed-shape models. Never silently downgrade a strict
        # request: doing so moves protocol failures from the relay to the graph.
        if strict:
            compatible = provider_strict_schema(schema)
            return await self._delegate.complete_json(system=system, user=user, schema=compatible,
                                                      model=model, temperature=temperature, strict=True)
        compatible_schema = provider_strict_schema(schema) if strict else schema
        return await self._delegate.complete_json(system=system, user=user, schema=compatible_schema,
                                                  model=model, temperature=temperature, strict=strict)

    async def aclose(self) -> None:
        close = getattr(self._delegate, "aclose", None)
        if close is not None:
            await close()
