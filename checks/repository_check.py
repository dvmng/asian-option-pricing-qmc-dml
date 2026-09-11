from __future__ import annotations

import hashlib
import json
from pathlib import Path


REQUIRED_MODEL_FILES = ("COMPLETED", "history.csv", "metadata.json", "model.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_hash(path: Path) -> tuple[str, str | None]:
    obj = load_json(path)
    expected = obj.pop("protocol_hash_sha256", None)
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), expected


def check_pricing_dataset(root: Path, errors: list[str]) -> None:
    data_dir = root / "data" / "frozen" / "pricing_dataset"
    manifest_path = data_dir / "MANIFEST.json"
    config_path = data_dir / "config.json"

    for path in (manifest_path, config_path):
        if not path.exists():
            errors.append(f"Missing pricing artifact: {path.relative_to(root)}")

    if not manifest_path.exists() or not config_path.exists():
        return

    manifest = load_json(manifest_path)
    expected_config_hash = manifest.get("config_sha256")
    if expected_config_hash and sha256(config_path) != expected_config_hash:
        errors.append("Pricing config hash mismatch: data/frozen/pricing_dataset/config.json")

    listed = manifest.get("files", {})
    for name in ("train.csv.gz", "val.csv.gz", "test.csv.gz", "reference.csv.gz"):
        path = data_dir / name
        if not path.exists():
            errors.append(f"Missing pricing split: data/frozen/pricing_dataset/{name}")
            continue
        expected = listed.get(name)
        if expected is None:
            errors.append(f"Pricing manifest does not list {name}.")
        elif sha256(path) != expected:
            errors.append(f"Pricing split hash mismatch: data/frozen/pricing_dataset/{name}")


def check_ml_protocol_and_models(root: Path, errors: list[str]) -> tuple[int, int | None]:
    protocol_path = root / "protocols" / "ml_protocol.json"
    if not protocol_path.exists():
        errors.append("Missing protocol: protocols/ml_protocol.json")
        return 0, None

    protocol = load_json(protocol_path)
    if protocol.get("status") != "FROZEN_BEFORE_COMPUTE_FINAL_TRAINING":
        errors.append("Unexpected status in protocols/ml_protocol.json.")

    sizes = [int(x) for x in protocol.get("final_train_sizes", [])]
    seeds = [int(x) for x in protocol.get("final_training_seeds", [])]
    parameter_count = protocol.get("expected_parameter_count")
    parameter_count = int(parameter_count) if parameter_count is not None else None

    if not sizes or not seeds:
        errors.append("Final training sizes or seeds are missing from ml_protocol.json.")
        return 0, parameter_count

    expected_names = {
        f"{model}_n{size}_seed{seed}"
        for size in sizes
        for seed in seeds
        for model in ("mlp", "dml")
    }
    models_dir = root / "models" / "final"
    if not models_dir.exists():
        errors.append("Missing directory: models/final")
        return len(expected_names), parameter_count

    actual_dirs = {p.name for p in models_dir.iterdir() if p.is_dir()}
    smoke = sorted(name for name in actual_dirs if name.startswith("SMOKE_"))
    if smoke:
        errors.append(f"Smoke models found in models/final: {smoke}")

    actual_final = {name for name in actual_dirs if not name.startswith("SMOKE_")}
    missing = sorted(expected_names - actual_final)
    extra = sorted(actual_final - expected_names)
    if missing:
        errors.append(f"Missing final model runs: {missing}")
    if extra:
        errors.append(f"Unexpected directories in models/final: {extra}")

    protocol_digest = sha256(protocol_path)
    for name in sorted(expected_names & actual_final):
        run_dir = models_dir / name
        for filename in REQUIRED_MODEL_FILES:
            if not (run_dir / filename).exists():
                errors.append(f"Missing model artifact: models/final/{name}/{filename}")

        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if metadata.get("run_name") != name:
            errors.append(f"Run-name mismatch in models/final/{name}/metadata.json")
        if metadata.get("frozen_protocol_sha256") != protocol_digest:
            errors.append(f"Protocol hash mismatch in models/final/{name}/metadata.json")
        if parameter_count is not None and int(metadata.get("parameter_count", -1)) != parameter_count:
            errors.append(f"Parameter-count mismatch in models/final/{name}/metadata.json")
        if metadata.get("no_early_stopping") is not True:
            errors.append(f"Early-stopping flag mismatch in models/final/{name}/metadata.json")
        if int(metadata.get("max_epochs_used", -1)) != int(protocol.get("final_max_epochs", -2)):
            errors.append(f"Epoch-budget mismatch in models/final/{name}/metadata.json")

    return len(expected_names), parameter_count


def check_hedge_protocols(root: Path, errors: list[str]) -> None:
    for rel in (
        "protocols/local_hedge_protocol.json",
        "protocols/forward_stress_protocol.json",
    ):
        path = root / rel
        if not path.exists():
            errors.append(f"Missing protocol: {rel}")
            continue
        actual, expected = protocol_hash(path)
        if expected is None:
            errors.append(f"Protocol hash field missing: {rel}")
        elif actual != expected:
            errors.append(f"Protocol hash mismatch: {rel}")


def check_final_results(
    root: Path,
    expected_runs: int,
    expected_parameter_count: int | None,
    errors: list[str],
) -> None:
    required_files = (
        "results/ml/final_evaluation/aggregate_metrics.csv",
        "results/ml/final_evaluation/per_run_metrics.csv",
        "results/ml/final_evaluation/paired_mlp_dml.csv",
        "results/hedging/local/final_local_hedge_metrics_pooled.csv",
        "results/hedging/dynamic_forward/final_forward_metrics_pooled.csv",
        "results/diagnostics/fixing_boundary/error_by_fixing_boundary_distance.csv",
        "thesis/main.tex",
        "thesis/main.pdf",
    )
    for rel in required_files:
        if not (root / rel).exists():
            errors.append(f"Missing final thesis artifact: {rel}")

    audit_path = root / "results" / "ml" / "final_evaluation" / "evaluation_audit.json"
    if audit_path.exists():
        audit = load_json(audit_path)
        if expected_runs and int(audit.get("n_completed_runs", -1)) != expected_runs:
            errors.append("Final evaluation audit has an unexpected completed-run count.")
        if expected_runs and int(audit.get("expected_runs", -1)) != expected_runs:
            errors.append("Final evaluation audit has an unexpected expected-run count.")
        if (
            expected_parameter_count is not None
            and int(audit.get("expected_parameter_count", -1)) != expected_parameter_count
        ):
            errors.append("Final evaluation audit has an unexpected parameter count.")


def run_checks(root: Path) -> None:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    required_dirs = (
        "src/compute",
        "src/compute_ml",
        "src/compute_hedging",
        "src/tfm_project",
        "experiments",
        "checks",
        "data/raw",
        "data/processed",
        "data/frozen/pricing_dataset",
        "data/frozen/ml_prepared",
        "data/frozen/model_selection_validation",
        "models/final",
        "results/ml/final_evaluation",
        "results/hedging/local",
        "results/hedging/dynamic_forward",
        "protocols",
        "thesis",
    )
    for rel in required_dirs:
        if not (root / rel).is_dir():
            errors.append(f"Missing required directory: {rel}")

    if (root / "src" / "compute" / "data").exists():
        errors.append("Data found inside src/compute/data; project data must live under data/.")

    for base in (root / "src", root / "experiments", root / "checks"):
        if not base.exists():
            continue
        if list(base.rglob("__pycache__")) or list(base.rglob("*.pyc")):
            warnings.append(f"Python cache files found under {base.relative_to(root)}.")

    archive = root / "archive"
    if archive.exists():
        unexpected = [p.name for p in archive.iterdir() if p.name != "recovered_scripts"]
        if unexpected:
            warnings.append(f"Unexpected development archive entries: {unexpected}")
        recovered = archive / "recovered_scripts"
        if recovered.exists():
            provenance = json.loads((recovered / "PROVENANCE.json").read_text(encoding="utf-8"))
            import hashlib
            for item in provenance:
                if hashlib.sha256((recovered / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
                    errors.append(f"Recovered script hash mismatch: {item['file']}")

    for obsolete in ("TFM_old", "MIGRATION_MANIFEST.json"):
        if (root / obsolete).exists():
            warnings.append(f"Development-only artifact still present: {obsolete}")

    check_pricing_dataset(root, errors)
    expected_runs, parameter_count = check_ml_protocol_and_models(root, errors)
    check_hedge_protocols(root, errors)
    check_final_results(root, expected_runs, parameter_count, errors)

    print("\nREPOSITORY CHECK")
    print("=" * 72)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"\nRepository validation failed with {len(errors)} error(s).")

    print("Frozen pricing dataset:    PASS")
    print("Final ML model grid:        PASS")
    print("Frozen hedge protocols:    PASS")
    print("Final result evidence:      PASS")
    print("Repository structure:       PASS")
    print("\nTFM REPOSITORY CHECK PASSED.")
