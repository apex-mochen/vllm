# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.structured_output.backend_guidance import (
    process_for_additional_properties,
)


def test_const_instance_data_is_not_mutated():
    """Object literals under ``const`` must be left untouched, even when they
    contain a ``properties`` member (vllm#58695)."""
    feature = {"type": "Feature", "geometry": None, "properties": {"name": "HQ"}}
    out = process_for_additional_properties({"const": feature})
    assert out == {"const": feature}


def test_enum_instance_data_is_not_mutated():
    feature = {"type": "Feature", "properties": {"name": "HQ"}}
    out = process_for_additional_properties({"enum": [feature]})
    assert out == {"enum": [feature]}


def test_schema_objects_still_get_additional_properties_false():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "object", "properties": {"x": {"type": "string"}}}},
    }
    out = process_for_additional_properties(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["a"]["additionalProperties"] is False


def test_schema_combinators_are_traversed():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
        ]
    }
    out = process_for_additional_properties(schema)
    assert out["anyOf"][0]["additionalProperties"] is False