"""Parse-check all modified source files. Delete after running."""
import os
import ast
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root, "src"))

files = [
    "train.py",
    "eval_final.py",
    "app.py",
    "src/__init__.py",
    "src/paths.py",
    "src/model.py",
    "src/dataset.py",
    "src/augmentations.py",
    "src/spatial_math.py",
    "src/predict.py",
    "src/generate_prompt.py",
    "src/generate_reports.py",
    "src/convert_sen12.py",
    "src/geo_lookup.py",
    "src/custom_vocabulary.py",
    "src/confusion_matrix.py",
    "src/probe_weights.py",
    "src/test_gpu.py",
]

failed = []
for rel in files:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        print(f"  MISSING  {rel}")
        failed.append((rel, "file not found"))
        continue
    try:
        with open(path, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK       {rel}")
    except Exception as exc:
        print(f"  FAIL     {rel}: {exc}")
        failed.append((rel, exc))

print()
if failed:
    print(f"{len(failed)} file(s) have issues:")
    for f, e in failed:
        print(f"  {f}: {e}")
    sys.exit(1)
else:
    print(f"All {len(files)} files parse cleanly.")
