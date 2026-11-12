"""Read-only checks of baseline immutability and new artifact integrity."""
from pathlib import Path
import csv
import hashlib
import json
import math
import py_compile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/xai/improvements/validation"


def validate():
    baseline = json.loads((OUT / "baseline_artifact_manifest.json").read_text())
    changed = []
    for item in baseline["artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file():
            changed.append({"path": item["path"], "reason": "missing"})
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                changed.append({"path": item["path"], "reason": "checksum mismatch"})
    checked = []
    errors = []
    roots = [ROOT / "results/xai/improvements", ROOT / "results/xai/final_iccad5/degeneracy_forensics"]
    for directory in roots:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.parent == OUT:
                continue
            try:
                if path.suffix == ".csv":
                    with path.open(newline="") as stream:
                        reader = csv.DictReader(stream)
                        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                            raise ValueError("missing or duplicate column names")
                        rows = 0
                        blanks = 0
                        for number, row in enumerate(reader, 2):
                            if None in row or any(v is None for v in row.values()):
                                raise ValueError(f"ragged row {number}")
                            for key, value in row.items():
                                if value == "":
                                    blanks += 1
                                    continue
                                try:
                                    numeric = float(value)
                                except ValueError:
                                    continue
                                if not math.isfinite(numeric):
                                    raise ValueError(f"nonfinite value at row {number}, {key}")
                            rows += 1
                    checked.append({"path": str(path.relative_to(ROOT)), "rows": rows,
                                    "blank_cells_requiring_semantic_review": blanks})
                elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    with Image.open(path) as image:
                        image.verify()
                    checked.append({"path": str(path.relative_to(ROOT)), "image": "verified"})
                elif path.suffix == ".json":
                    json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                    checked.append({"path": str(path.relative_to(ROOT)), "json": "parsed"})
            except Exception as exc:
                errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
    scripts = sorted((ROOT / "scripts/improvements").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    for path in scripts:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
    report = {"baseline_files_checked": len(baseline["artifacts"]),
              "baseline_checkpoints_checked": sum(r["path"].endswith(".keras") for r in baseline["artifacts"]),
              "baseline_changes": changed, "new_artifacts_checked": checked,
              "syntax_checked": [str(p.relative_to(ROOT)) for p in scripts], "errors": errors,
              "limitations": "Syntax is not runtime verification; blank cells require validity/reason review. GPU execution is not certified by this validator."}
    (OUT / "artifact_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["baseline_files_checked", "baseline_checkpoints_checked", "baseline_changes", "errors"]}, indent=2))
    print("New artifacts checked:", len(checked), "scripts compiled:", len(scripts))
    return not changed and not errors


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
