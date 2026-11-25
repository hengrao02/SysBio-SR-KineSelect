from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INITIAL_CONDITION_SOURCE = "trajectory_first_observation"


def _bitwise_equal_float64(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left, dtype=np.float64)
    right_array = np.ascontiguousarray(right, dtype=np.float64)
    return (
        left_array.shape == right_array.shape
        and left_array.tobytes() == right_array.tobytes()
    )


@dataclass
class SysBioSRTrajectory:
    name: str
    file: Path
    times: np.ndarray
    values: np.ndarray
    initial_condition: np.ndarray

    def rollout_initial_condition(self) -> np.ndarray:
        """Return the only initial condition permitted for a scored rollout."""
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(f"trajectory {self.name!r} has no first observation")
        return values[0].copy()

    def initial_condition_audit(self) -> dict[str, Any]:
        """Compare provenance metadata with the authoritative first observation."""
        observed = self.rollout_initial_condition()
        declared = np.asarray(self.initial_condition, dtype=np.float64).reshape(-1)
        same_shape = declared.shape == observed.shape
        max_abs_diff = None
        if same_shape and declared.size:
            max_abs_diff = float(np.max(np.abs(declared - observed)))
        return {
            "initial_condition_source": INITIAL_CONDITION_SOURCE,
            "metadata_initial_condition_present": bool(declared.size),
            "metadata_initial_condition_bitwise_match": (
                same_shape and _bitwise_equal_float64(declared, observed)
            ),
            "metadata_initial_condition_max_abs_diff": max_abs_diff,
        }

    def assert_rollout_initial_condition(self, y0: np.ndarray) -> None:
        """Fail closed unless ``y0`` is bitwise identical to the target first row."""
        observed = self.rollout_initial_condition()
        candidate = np.asarray(y0, dtype=np.float64).reshape(-1)
        if not _bitwise_equal_float64(candidate, observed):
            raise AssertionError(
                f"formal rollout y0 for trajectory {self.name!r} must be bitwise "
                "identical to trajectory.values[0]"
            )


@dataclass
class SysBioSRSystem:
    system_dir: Path
    system_id: str
    state_names: list[str]
    const_names: list[str]
    default_consts: np.ndarray
    trajectories: dict[str, SysBioSRTrajectory]
    metadata: dict[str, Any]
    trajectory_specs: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False
    )
    equation_path: Path | None = None
    annotation_path: Path | None = None
    xml_path: Path | None = None

    def get_trajectory(self, name: str = "default") -> SysBioSRTrajectory:
        if name in self.trajectories:
            return self.trajectories[name]
        if name not in self.trajectory_specs:
            available = sorted(set(self.trajectories) | set(self.trajectory_specs))
            raise KeyError(f"Trajectory '{name}' not found. Available: {available}")

        spec = self.trajectory_specs[name]
        csv_path = Path(spec["file"])
        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            raise ValueError(f"trajectory csv must contain 'time' column: {csv_path}")
        missing_states = [
            state for state in self.state_names if state not in df.columns
        ]
        if missing_states:
            raise ValueError(
                f"trajectory csv missing states {missing_states}: {csv_path}"
            )
        trajectory = SysBioSRTrajectory(
            name=name,
            file=csv_path,
            times=df["time"].to_numpy(dtype=float),
            values=df[self.state_names].to_numpy(dtype=float),
            initial_condition=np.asarray(
                spec.get("initial_condition", []), dtype=float
            ),
        )
        self.trajectories[name] = trajectory
        return trajectory

    def get_fit_inputs(
        self, trajectory_name: str = "default"
    ) -> tuple[np.ndarray, np.ndarray]:
        traj = self.get_trajectory(trajectory_name)
        return traj.times.copy(), traj.values.copy()

    def get_multi_fit_inputs(
        self, trajectory_names: list[str] | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if trajectory_names is None:
            trajectory_names = list(self.trajectory_specs or self.trajectories)
        times, trajectories = [], []
        for name in trajectory_names:
            traj = self.get_trajectory(name)
            times.append(traj.times.copy())
            trajectories.append(traj.values.copy())
        return times, trajectories

    def load_evaluation_view(self) -> dict[str, Any]:
        """Load ground truth only after fitting has completed.

        Fit code receives trajectory arrays and state names, not this view.
        Keeping the read explicit prevents accidental annotation/equation
        leakage into candidate construction or hyperparameter selection.
        """
        equation = None
        annotation = None
        if self.equation_path is not None:
            equation = json.loads(self.equation_path.read_text(encoding="utf-8"))
        if self.annotation_path is not None:
            annotation = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        return {"equation": equation, "reaction_annotation": annotation}

    @classmethod
    def from_dir(
        cls,
        system_dir: str | Path,
        *,
        preload_trajectories: bool = True,
    ) -> "SysBioSRSystem":
        system_dir = Path(system_dir).resolve()
        traj_dir = system_dir / "traj"
        metadata_path = traj_dir / "trajectory_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"trajectory metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state_names = list(metadata["state_names"])
        const_names = list(metadata.get("const_names", []))
        default_consts = np.asarray(metadata.get("default_consts", []), dtype=float)

        trajectory_specs: dict[str, dict[str, Any]] = {}
        for traj_info in metadata.get("trajectories", []):
            name = str(traj_info["name"])
            csv_path = traj_dir / traj_info["file"]
            if not csv_path.exists():
                raise FileNotFoundError(f"trajectory csv not found: {csv_path}")
            if name in trajectory_specs:
                raise ValueError(f"duplicate trajectory name {name!r}: {metadata_path}")
            trajectory_specs[name] = {
                "file": csv_path,
                "initial_condition": list(traj_info.get("initial_condition", [])),
            }

        xml_candidates = sorted(system_dir.glob("*.xml"))
        equation_path = system_dir / "equation.json"
        annotation_path = system_dir / "reaction_annotation.json"

        system = cls(
            system_dir=system_dir,
            system_id=str(metadata.get("system_id", system_dir.name)),
            state_names=state_names,
            const_names=const_names,
            default_consts=default_consts,
            trajectories={},
            metadata=metadata,
            trajectory_specs=trajectory_specs,
            equation_path=equation_path if equation_path.exists() else None,
            annotation_path=annotation_path if annotation_path.exists() else None,
            xml_path=xml_candidates[0] if xml_candidates else None,
        )
        if preload_trajectories:
            for name in trajectory_specs:
                system.get_trajectory(name)
        return system


class BenchmarkIntegrityError(RuntimeError):
    """Raised when a benchmark partition disagrees with its manifest."""


PUBLISHED_INTEGRITY_MIGRATION = (
    "This configuration requests published-data integrity verification, which is not "
    "included in this distribution. Ordinary runs verify benchmark membership and actual "
    "inputs. Remove verify_data_integrity / verify_content from the runtime configuration."
)


def benchmark_root(dataset_root: str | Path) -> Path:
    """Accept either the dataset directory or its benchmark child."""
    root = Path(dataset_root).expanduser().resolve()
    return root / "benchmark" if (root / "benchmark").is_dir() else root


def benchmark_definition_path(dataset_root: str | Path) -> Path:
    """Locate the sole public definition in the supplied data tree."""
    return benchmark_root(dataset_root) / "benchmark.yaml"


def reject_runtime_integrity_request(requested: bool) -> None:
    if requested:
        raise BenchmarkIntegrityError(PUBLISHED_INTEGRITY_MIGRATION)


@dataclass(frozen=True)
class BenchmarkSystemRow:
    system_id: str
    benchmark_partition: str
    dataset_version: str
    reaction_metrics: bool
    initial_condition_source: str = INITIAL_CONDITION_SOURCE


class BenchmarkDefinition:
    """Read the public system list independently of build and audit records."""

    def __init__(self, path: str | Path) -> None:
        import yaml

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"benchmark definition not found: {self.path}")
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "name",
            "version",
            "protocol",
            "partitions",
        }:
            raise BenchmarkIntegrityError(
                "definition requires name, version, protocol and partitions"
            )
        if (
            payload["name"] != "SysBioSR"
            or not isinstance(payload["version"], str)
            or not payload["version"]
        ):
            raise BenchmarkIntegrityError("invalid benchmark name/version")
        expected_protocol = {
            "fit_trajectory": "default",
            "evaluation_trajectory": "perturbed",
            "initial_condition": "first_observation",
        }
        if payload["protocol"] != expected_protocol:
            raise BenchmarkIntegrityError("unsupported benchmark protocol")
        groups = payload["partitions"]
        if not isinstance(groups, dict) or set(groups) != {
            "main",
            "challenge",
            "stress",
        }:
            raise BenchmarkIntegrityError(
                "definition requires main/challenge/stress partitions"
            )
        self.spec = payload
        self.dataset_version = payload["version"]
        self.protocol = dict(payload["protocol"])
        self.rows: list[BenchmarkSystemRow] = []
        seen = set()
        for partition in ("main", "challenge", "stress"):
            group = groups[partition]
            if not isinstance(group, dict) or set(group) != {
                "reaction_metrics",
                "systems",
            }:
                raise BenchmarkIntegrityError(
                    f"invalid partition definition: {partition}"
                )
            if (
                not isinstance(group["reaction_metrics"], bool)
                or not isinstance(group["systems"], list)
                or not group["systems"]
            ):
                raise BenchmarkIntegrityError(
                    f"invalid systems/metric policy: {partition}"
                )
            if group["reaction_metrics"] != (partition != "stress"):
                raise BenchmarkIntegrityError(
                    "reaction metrics apply to main/challenge only"
                )
            for sid in group["systems"]:
                if (
                    not isinstance(sid, str)
                    or not sid
                    or sid in seen
                    or sid in {".", ".."}
                    or "/" in sid
                    or "\\" in sid
                ):
                    raise BenchmarkIntegrityError(
                        f"invalid or duplicate system ID: {sid!r}"
                    )
                seen.add(sid)
                self.rows.append(
                    BenchmarkSystemRow(
                        sid, partition, self.dataset_version, group["reaction_metrics"]
                    )
                )
        self.release_context = {
            "dataset_version": self.dataset_version,
            "benchmark_spec": self.spec,
        }

    def partition_rows(self, partition: str) -> list[BenchmarkSystemRow]:
        return [row for row in self.rows if row.benchmark_partition == partition]

    def get_row(
        self, system_id: str, *, partition: str | None = None
    ) -> BenchmarkSystemRow:
        matches = [
            row
            for row in self.rows
            if row.system_id == system_id
            and (partition is None or row.benchmark_partition == partition)
        ]
        if len(matches) != 1:
            raise BenchmarkIntegrityError(
                f"definition must contain exactly one {system_id!r} in {partition!r}"
            )
        return matches[0]

    def validate_system(
        self,
        system_dir: str | Path,
        *,
        partition: str | None = None,
        verify_content: bool = False,
    ) -> BenchmarkSystemRow:

        reject_runtime_integrity_request(verify_content)
        path = Path(system_dir).resolve()
        row = self.get_row(path.name, partition=partition)
        if not (path / "traj/trajectory_metadata.json").is_file():
            raise BenchmarkIntegrityError(
                f"trajectory metadata missing for {row.system_id}"
            )
        return row

    def validate_partition(
        self, systems_root: str | Path, *, partition: str, verify_content: bool = False
    ) -> list[Path]:
        reject_runtime_integrity_request(verify_content)
        root = Path(systems_root).resolve()
        expected = {row.system_id for row in self.partition_rows(partition)}
        if not expected:
            raise BenchmarkIntegrityError(f"unknown or empty partition: {partition}")
        actual = {
            p.name: p.resolve()
            for p in root.iterdir()
            if p.is_dir() and (p / "traj/trajectory_metadata.json").is_file()
        }
        if set(actual) != expected:
            raise BenchmarkIntegrityError(
                f"partition {partition} differs from definition; missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)}"
            )
        for sid in sorted(actual):
            self.validate_system(
                actual[sid], partition=partition, verify_content=verify_content
            )
        return [actual[sid] for sid in sorted(actual)]


def validate_system_preflight(
    manifest_path: str | Path,
    system_dir: str | Path,
    *,
    partition: str,
    verify_content: bool = False,
) -> tuple[BenchmarkSystemRow, dict[str, str | None]]:
    """Run manifest/system integrity checks before the isolated fit phase."""
    manifest = BenchmarkDefinition(manifest_path)
    row = manifest.validate_system(
        system_dir,
        partition=partition,
        verify_content=verify_content,
    )
    return row, dict(manifest.release_context)


def validate_partition_preflight(
    manifest_path: str | Path,
    systems_root: str | Path,
    *,
    partition: str,
    verify_content: bool = False,
) -> tuple[list[Path], list[BenchmarkSystemRow], dict[str, str | None]]:
    """Validate a whole partition in a process isolated from method fitting."""
    manifest = BenchmarkDefinition(manifest_path)
    paths = manifest.validate_partition(
        systems_root,
        partition=partition,
        verify_content=verify_content,
    )
    return paths, manifest.partition_rows(partition), dict(manifest.release_context)
