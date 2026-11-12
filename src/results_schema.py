"""Workstream D: Machine-Readable Results Schemas and Validation.

Defines schemas and validators for:
  1. ExperimentRunManifest
  2. ThresholdAnalysisSummary
  3. MultiSeedSummary
  4. XAIEvaluationSummary
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunManifestSchema:
    contract_version: str
    experiment_name: str
    timestamp: str
    hyperparameters: Dict[str, Any]
    dataset_counts: Dict[str, Dict[str, int]]
    manifest_hashes: Dict[str, str]
    checkpoint: Dict[str, Any]
    metrics_val_default: Dict[str, Any]
    metrics_clean_test_default: Dict[str, Any]
    metrics_clean_test_calibrated: Dict[str, Any]
    training_duration_seconds: float
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifestSchema":
        return cls(
            contract_version=data["contract_version"],
            experiment_name=data["experiment_name"],
            timestamp=data.get("timestamp", ""),
            hyperparameters=data.get("hyperparameters", {}),
            dataset_counts=data.get("dataset_counts", {}),
            manifest_hashes=data.get("manifest_hashes", {}),
            checkpoint=data.get("checkpoint", {}),
            metrics_val_default=data.get("metrics_val_default", {}),
            metrics_clean_test_default=data.get("metrics_clean_test_default", {}),
            metrics_clean_test_calibrated=data.get("metrics_clean_test_calibrated", {}),
            training_duration_seconds=data.get("training_duration_seconds", 0.0),
            is_synthetic=data.get("is_synthetic", False),
        )


@dataclass
class ThresholdAnalysisSchema:
    experiment_name: str
    calibrated_threshold: float
    objective: str
    validation_metrics_calibrated: Dict[str, Any]
    test_metrics: Optional[Dict[str, Any]] = None
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ThresholdAnalysisSchema":
        return cls(
            experiment_name=data["experiment_name"],
            calibrated_threshold=float(data["calibrated_threshold"]),
            objective=data["objective"],
            validation_metrics_calibrated=data["validation_metrics_calibrated"],
            test_metrics=data.get("test_metrics"),
            is_synthetic=data.get("is_synthetic", False),
        )


@dataclass
class MultiSeedSummarySchema:
    strategy: str
    seeds: List[int]
    aggregate_statistics: Dict[str, Any]
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "MultiSeedSummarySchema":
        return cls(
            strategy=data["strategy"],
            seeds=data["seeds"],
            aggregate_statistics=data["aggregate_statistics"],
            is_synthetic=data.get("is_synthetic", False),
        )


@dataclass
class XAIEvaluationSchema:
    contract_version: str
    methods_evaluated: List[str]
    method_summary: Dict[str, Dict[str, float]]
    quadrants_represented: List[str]
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "XAIEvaluationSchema":
        return cls(
            contract_version=data["contract_version"],
            methods_evaluated=data["methods_evaluated"],
            method_summary=data["method_summary"],
            quadrants_represented=data.get("quadrants_represented", []),
            is_synthetic=data.get("is_synthetic", False),
        )


def validate_schema(data: dict, schema_type: str) -> bool:
    """Validate raw dictionary against schema."""
    try:
        if schema_type == "run_manifest":
            RunManifestSchema.from_dict(data)
        elif schema_type == "threshold":
            ThresholdAnalysisSchema.from_dict(data)
        elif schema_type == "multiseed":
            MultiSeedSummarySchema.from_dict(data)
        elif schema_type == "xai":
            XAIEvaluationSchema.from_dict(data)
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")
        return True
    except Exception as e:
        print(f"Validation error for {schema_type}: {e}")
        return False
