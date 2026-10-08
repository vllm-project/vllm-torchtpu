"""
test_json_recorder.py
---------------------
Unit tests for JsonResultRecorder, CompositeResultRecorder, and JSON schema validation.
"""

import json
import os

import jsonschema
import pytest
from core.csv_recorder import CsvResultRecorder
from core.json_recorder import CompositeResultRecorder, JsonResultRecorder


@pytest.fixture
def schema_path():
    """Returns the path to the canonical sweep JSON schema."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    schema_file = os.path.join(tests_dir, "schemas", "sweep.schema.json")
    if not os.path.exists(schema_file):
        # Fallback to experiments/schemas/sweep.schema.json
        schema_file = os.path.abspath(
            os.path.join(tests_dir, "..", "schemas", "sweep.schema.json"))
    assert os.path.exists(
        schema_file), f"Schema file not found at {schema_file}"
    return schema_file


@pytest.fixture
def loaded_schema(schema_path):
    with open(schema_path, "r") as f:
        return json.load(f)


def test_json_recorder_initialization_and_schema(tmp_path, loaded_schema):
    """Tests fresh sweep.json creation and strict schema adherence."""
    json_file = str(tmp_path / "sweep.json")
    metadata = {
        "description": "Test sweep for validation",
        "mode": "profile",
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "sweep_matrix": {
            "batch_size": [1, 2],
            "input_len": [128, 256],
        },
    }

    _ = JsonResultRecorder(
        json_file=json_file,
        sweep_id="20260828_120000",
        metadata=metadata,
        user_ldap="test_user",
        hardware={
            "accelerator_type": "TPU v6e",
            "chip_count": 4
        },
    )

    assert os.path.exists(json_file)
    with open(json_file, "r") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0.0"
    assert data["sweep_id"] == "20260828_120000"
    assert data["user_ldap"] == "test_user"
    assert data["status"] == "RUNNING"
    assert data["metadata"]["description"] == "Test sweep for validation"
    assert data["experiments"] == []

    # Validate against JSON schema
    jsonschema.validate(instance=data, schema=loaded_schema)


@pytest.mark.parametrize("sweep_id", [
    "20260828_120000",
    "20260828_120000_resume",
    "test_resume_json_exp",
    "custom-Named.Exp_01",
])
def test_json_recorder_accepts_custom_sweep_ids(tmp_path, loaded_schema,
                                                sweep_id):
    """Custom --experiment-id values must validate against the schema."""
    json_file = str(tmp_path / "sweep.json")
    JsonResultRecorder(
        json_file=json_file,
        sweep_id=sweep_id,
        metadata={
            "description": "custom id",
            "mode": "eval"
        },
    )
    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["sweep_id"] == sweep_id
    jsonschema.validate(instance=data, schema=loaded_schema)


@pytest.mark.parametrize("sweep_id", ["", " spaced id", "slash/id", "-lead"])
def test_sweep_id_schema_rejects_unsafe_ids(loaded_schema, sweep_id):
    """Path-unsafe identifiers are still rejected by the schema pattern."""
    data = {
        "schema_version": "1.0.0",
        "sweep_id": sweep_id,
        "created_at": "2026-08-28T12:00:00Z",
        "status": "RUNNING",
        "user_ldap": "user",
        "metadata": {
            "description": "",
            "mode": "eval"
        },
        "experiments": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=loaded_schema)


def test_json_recorder_records_trials_with_non_primitives(
        tmp_path, loaded_schema):
    """Tests recording trials with sets/tuples and non-primitive types."""
    json_file = str(tmp_path / "sweep.json")
    metadata = {
        "description": "Multi-mode benchmark sweep",
        "mode": "both",
        "sweep_matrix": {
            "model": ["test-model"]
        },
    }

    recorder = JsonResultRecorder(
        json_file=json_file,
        sweep_id="20260828_120000",
        metadata=metadata,
        hardware={
            "accelerator_type": "TPU v6e",
            "chip_count": 4
        },
    )

    # Record trial with set and tuple in parameters to verify fallback serialization
    trial1_params = {
        "model": "google/gemma-4-31B-it",
        "batch_size": 1,
        "input_len": 8192,
        "output_len": 1024,
        "consumed_keys": {"model", "batch_size"},  # Set!
        "shape_tuple": (8192, 1024),  # Tuple!
    }
    trial1_metrics = {
        "Throughput_tok_s": 58.62,
        "Duration_s": 17.4673,
        "total_tokens": 1024,
    }
    recorder.record(
        metadata=trial1_params,
        metrics=trial1_metrics,
        reproduction_cmd=
        "python3 run_sweep_orchestrator.py --config config.yaml",
    )

    recorder.finalize("COMPLETED")

    with open(json_file, "r") as f:
        data = json.load(f)

    assert data["status"] == "COMPLETED"
    assert len(data["experiments"]) == 1
    exp0 = data["experiments"][0]
    assert isinstance(exp0["parameters"]["consumed_keys"], list)
    assert isinstance(exp0["parameters"]["shape_tuple"], list)

    # Strictly validate against the official schema
    jsonschema.validate(instance=data, schema=loaded_schema)


def test_json_recorder_resumption(tmp_path, loaded_schema):
    """Tests that resuming an existing sweep appends correctly."""
    json_file = str(tmp_path / "sweep.json")
    recorder = JsonResultRecorder(
        json_file=json_file,
        sweep_id="20260828_120000",
        metadata={
            "description": "Original run",
            "mode": "profile"
        },
    )
    recorder.record({"batch_size": 1}, {"Throughput_tok_s": 40.0})

    # Simulate process restart / resume
    resume_recorder = JsonResultRecorder(
        json_file=json_file,
        sweep_id="20260828_120000",
    )
    assert len(resume_recorder.data["experiments"]) == 1
    assert resume_recorder.data["status"] == "RUNNING"

    resume_recorder.record({"batch_size": 2}, {"Throughput_tok_s": 75.0})
    resume_recorder.finalize("COMPLETED")

    with open(json_file, "r") as f:
        data = json.load(f)

    assert len(data["experiments"]) == 2
    assert data["experiments"][1]["trial_index"] == 1
    jsonschema.validate(instance=data, schema=loaded_schema)


def test_composite_result_recorder(tmp_path, loaded_schema):
    """Tests CompositeResultRecorder updates both CSV and JSON in sync."""
    csv_file = str(tmp_path / "results.csv")
    json_file = str(tmp_path / "sweep.json")

    csv_rec = CsvResultRecorder(csv_file)
    json_rec = JsonResultRecorder(
        json_file=json_file,
        sweep_id="20260828_120000",
        metadata={
            "description": "Composite test",
            "mode": "profile"
        },
    )
    composite = CompositeResultRecorder([csv_rec, json_rec])

    meta = {"model": "test", "batch_size": 4}
    metrics = {"Throughput_tok_s": 100.0, "Duration_s": 5.0}
    composite.record(meta, metrics, reproduction_cmd="cmd --test")
    composite.finalize("COMPLETED")

    # Verify CSV
    assert os.path.exists(csv_file)
    with open(csv_file, "r") as f:
        csv_content = f.read()
    assert "Throughput_tok_s" in csv_content
    assert "100.0" in csv_content

    # Verify JSON
    assert os.path.exists(json_file)
    with open(json_file, "r") as f:
        json_data = json.load(f)
    assert len(json_data["experiments"]) == 1
    assert json_data["status"] == "COMPLETED"
    jsonschema.validate(instance=json_data, schema=loaded_schema)
