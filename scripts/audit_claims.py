#!/usr/bin/env python3
"""
scripts/audit_claims.py

Automated Claims & Provenance Auditor for SatQuery AI (SIH 26167).
Enforces Rule 1: Real data only.
Enforces Rule 2: Every percentage and metric claim in README, frontend, and active docs
                 must be backed by an executed report in reports/**/*.json
                 or published baselines in docs/published_baselines.json.
                 Each quoted number must cite file#key and match the verified value.
Enforces Rule 5: No hardcoded confidence literals.

Exit code:
0: All checks pass (no unbacked numbers, no fake models, no hardcoded confidences).
1: Violations found.
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_SRC_DIR = ROOT_DIR / "frontend" / "src"
README_FILE = ROOT_DIR / "README.md"
PUBLISHED_BASELINES_FILE = DOCS_DIR / "published_baselines.json"

# Historical archive documents that record past audits and are explicitly marked superseded
ARCHIVED_DOCS = {
    "FINAL_POSTER_CORRECTIONS.md",
    "FINAL_PROJECT_AUDIT.md",
    "MODEL_AUDIT_REPORT.md",
    "MODEL_AUDIT_FACTS.json",
    "PROJECT_CONTEXT.md",
    "VIDEO_PITCH_SCRIPT.md",
    "SIH_2026_PRESENTATION_MASTER_DECK.txt"
}

def extract_all_numbers_from_json(obj):
    """Recursively extract all numeric scalars and percentage interpretations from JSON."""
    nums = set()
    if isinstance(obj, dict):
        for v in obj.values():
            nums.update(extract_all_numbers_from_json(v))
    elif isinstance(obj, list):
        for v in obj:
            nums.update(extract_all_numbers_from_json(v))
    elif isinstance(obj, (int, float)):
        val = float(obj)
        nums.add(round(val, 2))
        nums.add(round(val, 1))
        nums.add(round(val, 4))
        if 0.0 <= val <= 1.0:
            nums.add(round(val * 100.0, 2))
            nums.add(round(val * 100.0, 1))
    return nums

def load_all_verified_report_numbers():
    """Load all numeric values from reports/**/*.json and docs/published_baselines.json."""
    all_numbers = set()
    oracle_reports = set()
    report_files = []
    
    # 1. reports/ directory
    if REPORTS_DIR.exists():
        for root, _, files in os.walk(REPORTS_DIR):
            for f in files:
                if f.endswith(".json"):
                    fpath = Path(root) / f
                    report_files.append(fpath)
                    try:
                        with open(fpath, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                            is_orc = False
                            if isinstance(data, dict):
                                is_orc = data.get("is_oracle") is True or (isinstance(data.get("results"), dict) and data["results"].get("is_oracle") is True)
                            if is_orc:
                                rel_p = str(fpath.relative_to(ROOT_DIR)).replace("\\", "/")
                                oracle_reports.add(rel_p)
                                oracle_reports.add(fpath.name)
                            all_numbers.update(extract_all_numbers_from_json(data))
                    except Exception as e:
                        print(f"[WARN] Failed to read report {fpath.name}: {e}")

    # 2. docs/published_baselines.json
    if PUBLISHED_BASELINES_FILE.exists():
        report_files.append(PUBLISHED_BASELINES_FILE)
        try:
            with open(PUBLISHED_BASELINES_FILE, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                all_numbers.update(extract_all_numbers_from_json(data))
        except Exception as e:
            print(f"[WARN] Failed to read published baselines: {e}")

    return all_numbers, report_files, oracle_reports

def resolve_json_pointer(file_rel_path: str, key_path: str):
    """
    Resolves file#key pointer.
    Example: reports/change.json#metrics.macro_change_miou_pct
    Example: docs/published_baselines.json#cdvqa_test1_resnet18_oa
    """
    fpath = ROOT_DIR / file_rel_path.replace("\\", "/")
    if not fpath.exists():
        return None, f"Cited file not found: {file_rel_path}"

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Failed to parse JSON in {file_rel_path}: {e}"

    parts = key_path.split(".")
    curr = data
    for p in parts:
        if isinstance(curr, dict) and p in curr:
            curr = curr[p]
        elif isinstance(curr, list) and p.isdigit() and int(p) < len(curr):
            curr = curr[int(p)]
        else:
            return None, f"Key segment '{p}' not found in {file_rel_path} (full key: {key_path})"

    # If the target object is a baseline entry dict containing 'value', resolve to that value
    if isinstance(curr, dict) and "value" in curr:
        curr = curr["value"]

    if isinstance(curr, (int, float)):
        return float(curr), None
    if isinstance(curr, str):
        cleaned = curr.replace("%", "").strip()
        try:
            return float(cleaned), None
        except ValueError:
            pass

    return None, f"Value at {file_rel_path}#{key_path} is non-numeric: {curr}"

def audit_hardcoded_confidences():
    """Fail if any literal `"confidence": 0.` remains in backend."""
    pattern = re.compile(r'["\']confidence["\']\s*:\s*0\.[0-9]+')
    violations = []

    for py_file in BACKEND_DIR.glob("**/*.py"):
        if "venv" in py_file.parts or ".pytest_cache" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    violations.append(f"{py_file.relative_to(ROOT_DIR)}:{idx}: {line.strip()}")
        except Exception as e:
            violations.append(f"Error reading {py_file}: {e}")

    return violations

def audit_health_endpoint():
    """Check backend/main.py /health for honest model names."""
    main_py = BACKEND_DIR / "main.py"
    if not main_py.exists():
        return ["backend/main.py does not exist"]

    content = main_py.read_text(encoding="utf-8")
    violations = []

    if "Florence-2-base (Fine-tuned" in content:
        violations.append("backend/main.py: /health still claims 'Florence-2-base (Fine-tuned...)' but actual code does not load it.")
    if "ResNet-18 Backbone" in content:
        violations.append("backend/main.py: /health still claims 'ResNet-18 Backbone' but actual model is a 482k-param custom U-Net.")

    return violations

def is_css_or_layout_line(line: str) -> bool:
    """Filter out CSS dimensions, Tailwind width/height classes, or comment modulos."""
    l = line.strip()
    if 'width="' in l or 'height="' in l or 'width:' in l or 'height:' in l:
        return True
    if 'w-[' in l or 'h-[' in l or 'top-[' in l or 'left-[' in l or 'right-[' in l or 'bottom-[' in l:
        return True
    if 'w-full' in l or 'h-full' in l:
        return True
    if '-10 % 3' in l or 'Positive modulo' in l:
        return True
    return False

def audit_text_percentages_and_citations(
    file_path: Path,
    content: str,
    verified_numbers: set,
    label: str,
    require_citations: bool = False,
    oracle_reports: set = None
):
    """
    Extracts every percentage and validates:
    1. Percentage exists in indexed numbers.
    2. If require_citations is True (enforced on README.md), EVERY percentage must have a matching file#key citation.
    3. If a file#key citation exists on a line with a percentage, validates the pointer and ensures the quoted value matches.
    4. Enforces rule: Any number or citation from an oracle report (is_oracle=true) CANNOT be cited as system or model accuracy.
    """
    violations = []
    pct_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%')
    citation_pattern = re.compile(r'([a-zA-Z0-9_\-\./\\]+\.json)#([a-zA-Z0-9_\-\.]+)')

    for idx, line in enumerate(content.splitlines(), start=1):
        if is_css_or_layout_line(line):
            continue

        pct_matches = pct_pattern.findall(line)
        if not pct_matches:
            continue
        cit_matches = citation_pattern.findall(line)

        # 0. Check for forbidden oracle citations or numbers claimed as system/model accuracy
        if oracle_reports and cit_matches:
            for rel_file, key in cit_matches:
                norm_rel = rel_file.replace("\\", "/")
                fname = Path(rel_file).name
                if norm_rel in oracle_reports or fname in oracle_reports:
                    system_claim_pattern = re.compile(r'(?:system|model|our|overall|satquery|pipeline)\s+accuracy', re.IGNORECASE)
                    if system_claim_pattern.search(line):
                        violations.append(
                            f"{label} [{file_path.name}:{idx}]: Oracle citation '{rel_file}#{key}' is FORBIDDEN from being cited as system accuracy! Line: '{line.strip()}'"
                        )

        # 1. Validate percentage existence in verified_numbers
        for m in pct_matches:
            val = float(m)
            if "." in m:
                decimals = len(m.split(".")[1])
                matched = round(val, decimals) in verified_numbers
            else:
                matched = int(val) in verified_numbers or round(val, 1) in verified_numbers or round(val, 2) in verified_numbers

            if not matched:
                violations.append(
                    f"{label} [{file_path.name}:{idx}]: Claimed {m}% has NO matching verified report entry in reports/*.json or docs/published_baselines.json! Line: '{line.strip()}'"
                )

        # 2. Resolve all citations present on this line
        resolved_citation_values = []
        if cit_matches:
            for rel_file, key in cit_matches:
                val, err = resolve_json_pointer(rel_file, key)
                if err:
                    violations.append(f"{label} [{file_path.name}:{idx}]: Broken citation '{rel_file}#{key}': {err}")
                else:
                    resolved_citation_values.append((rel_file, key, val))

        # 3. If citations are required (e.g. README.md), EVERY percentage on the line must match a citation
        if require_citations and pct_matches:
            if not resolved_citation_values:
                for m in pct_matches:
                    violations.append(
                        f"{label} [{file_path.name}:{idx}]: Quoted percentage {m}% is missing mandatory citation (file#key)! Line: '{line.strip()}'"
                    )
            else:
                for m in pct_matches:
                    target = float(m)
                    matched_cit = False
                    for rel_file, key, val in resolved_citation_values:
                        val_pct = val * 100.0 if (0.0 <= val <= 1.0 and target > 1.0) else val
                        if abs(target - val) < 0.05 or abs(target - val_pct) < 0.05:
                            matched_cit = True
                            break
                    if not matched_cit:
                        violations.append(
                            f"{label} [{file_path.name}:{idx}]: Quoted percentage {m}% has no matching citation among {[f'{r}#{k}' for r, k, _ in resolved_citation_values]}! Line: '{line.strip()}'"
                        )
        elif cit_matches and pct_matches:
            # If citations are present on non-README lines, ensure each citation matches at least one percentage on the line
            target_vals = [float(p) for p in pct_matches]
            for rel_file, key, val in resolved_citation_values:
                val_pct = val * 100.0 if (0.0 <= val <= 1.0 and max(target_vals) > 1.0) else val
                matched = any(abs(target - val) < 0.05 or abs(target - val_pct) < 0.05 for target in target_vals)
                if not matched:
                    violations.append(
                        f"{label} [{file_path.name}:{idx}]: Value mismatch for '{rel_file}#{key}'. JSON value={val} (pct={val_pct:.2f}), but quoted line has percentages: {pct_matches}"
                    )

    return violations

def audit_frontend_claims(verified_numbers: set):
    """Check frontend copy for unbacked metric percentages or forbidden phrases."""
    violations = []
    forbidden_claims = [
        (re.compile(r'90\.2%?\s*Accuracy', re.IGNORECASE), "Unverified 90.2% Accuracy claim"),
        (re.compile(r'Zero\s+FPR', re.IGNORECASE), "Unverified Zero FPR claim"),
        (re.compile(r'millimeter\s+precision', re.IGNORECASE), "Unverified millimeter precision claim"),
        (re.compile(r'<10ms\s+ONNX', re.IGNORECASE), "Unverified <10ms claim in landing page copy"),
        (re.compile(r'77\.82%?\s*mIoU', re.IGNORECASE), "Unverified 77.82% mIoU claim"),
        (re.compile(r'71\.30?%?\s*mIoU', re.IGNORECASE), "Unverified 71.3% mIoU claim"),
        (re.compile(r'75\.80?%?\s*mIoU', re.IGNORECASE), "Unverified 75.8% mIoU claim"),
    ]

    for ext in ["*.tsx", "*.ts", "*.jsx", "*.js"]:
        for code_file in FRONTEND_SRC_DIR.glob(f"**/{ext}"):
            if "node_modules" in code_file.parts or ".next" in code_file.parts:
                continue
            try:
                content = code_file.read_text(encoding="utf-8")
                for pattern, msg in forbidden_claims:
                    for idx, line in enumerate(content.splitlines(), start=1):
                        if pattern.search(line):
                            violations.append(f"Frontend [{code_file.name}:{idx}]: {msg} -> '{line.strip()}'")

                pct_violations = audit_text_percentages_and_citations(
                    code_file, content, verified_numbers, "Frontend"
                )
                violations.extend(pct_violations)
            except Exception as e:
                violations.append(f"Error reading {code_file}: {e}")

    return violations

def audit_readme_and_active_docs(verified_numbers: set, oracle_reports: set = None):
    """Check README.md and active docs/*.md for unbacked numbers, broken file#key citations, or unverified claims."""
    violations = []
    files_to_check = [README_FILE]
    if DOCS_DIR.exists():
        for doc in DOCS_DIR.glob("*.md"):
            if doc.name not in ARCHIVED_DOCS:
                files_to_check.append(doc)

    for doc_file in files_to_check:
        if not doc_file.exists():
            continue
        try:
            content = doc_file.read_text(encoding="utf-8")
            is_readme = (doc_file == README_FILE)
            pct_violations = audit_text_percentages_and_citations(
                doc_file, content, verified_numbers, doc_file.name, require_citations=is_readme, oracle_reports=oracle_reports
            )
            violations.extend(pct_violations)
        except Exception as e:
            violations.append(f"Error reading {doc_file}: {e}")

    return violations

def test_deliberate_wrong_number(verified_numbers: set):
    """
    Test that audit_claims fails when given a deliberately fabricated percentage.
    Required by Ground Rule 2.
    """
    print("[TEST] Running self-test against deliberately fabricated percentage '98.76%'...")
    test_fake_content = "This model achieves 98.76% accuracy on held-out data."
    dummy_path = Path("fake_document.md")
    fake_violations = audit_text_percentages_and_citations(dummy_path, test_fake_content, verified_numbers, "SelfTest")
    if len(fake_violations) >= 1 and any("98.76%" in v for v in fake_violations):
        print("[TEST PASS] Successfully caught deliberate fake number '98.76%' with violation:")
        print(f"            --> {fake_violations[0]}")
        return True
    else:
        print(f"[TEST FAIL] Failed to catch fake number. Violations caught: {fake_violations}")
        return False

def test_deliberate_broken_citation():
    """
    Test that audit_claims fails when given a broken file#key citation on a claimed number.
    Required by Ground Rule 2.
    """
    print("[TEST] Running self-test against deliberately broken citation 'reports/change.json#fake_metric_key'...")
    test_fake_content = "CDVQA achieves 50.0% change IoU reports/change.json#fake_metric_key"
    dummy_path = Path("fake_document.md")
    fake_violations = audit_text_percentages_and_citations(dummy_path, test_fake_content, {50.0}, "SelfTest")
    if len(fake_violations) >= 1 and any("Broken citation" in v for v in fake_violations):
        print("[TEST PASS] Successfully caught broken citation with violation:")
        print(f"            --> {fake_violations[0]}")
        return True
    else:
        print(f"[TEST FAIL] Failed to catch broken citation. Violations caught: {fake_violations}")
        return False

def test_deliberate_uncited_percentage_in_readme(verified_numbers: set):
    """
    Test that audit_claims fails when README has an uncited percentage.
    Required by Item 5.
    """
    print("[TEST] Running self-test against uncited percentage in README...")
    test_fake_content = "| Test Item | Metric | 67.71% accuracy on Test 1 |"
    dummy_path = Path("README.md")
    fake_violations = audit_text_percentages_and_citations(dummy_path, test_fake_content, verified_numbers, "SelfTest", require_citations=True)
    if len(fake_violations) >= 1 and any("missing mandatory citation" in v for v in fake_violations):
        print("[TEST PASS] Successfully caught uncited percentage in README with violation:")
        print(f"            --> {fake_violations[0]}")
        return True
    else:
        print(f"[TEST FAIL] Failed to catch uncited percentage in README. Violations caught: {fake_violations}")
        return False

def test_deliberate_oracle_citation_as_system_accuracy(oracle_reports: set):
    """
    Test that audit_claims fails when an oracle result is cited as system accuracy.
    Required by Item 2.
    """
    print("[TEST] Running self-test against deliberate oracle citation as system accuracy...")
    test_fake_content = "SatQuery AI achieves 99.94% system accuracy reports/benchmark_evaluation/cdvqa_gt_oracle_analysis.json#results.overall_accuracy_pct"
    dummy_path = Path("README.md")
    fake_violations = audit_text_percentages_and_citations(dummy_path, test_fake_content, {99.94}, "SelfTest", require_citations=True, oracle_reports=oracle_reports)
    if any("is FORBIDDEN from being cited as system accuracy" in v for v in fake_violations):
        print("[TEST PASS] Successfully caught deliberate oracle claim with violation:")
        print(f"            --> {fake_violations[0]}")
        return True
    else:
        print(f"[TEST FAIL] Failed to catch oracle claim. Violations: {fake_violations}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SatQuery Claims & Provenance Auditor")
    parser.add_argument("--test-wrong-claim", action="store_true", help="Run self-test with deliberate wrong number and citation")
    args = parser.parse_args()

    print("=" * 70)
    print("SatQuery AI: Strict Claims & Provenance Auditor (scripts/audit_claims.py)")
    print("=" * 70)

    verified_numbers, report_files, oracle_reports = load_all_verified_report_numbers()
    print(f"[INFO] Indexed {len(report_files)} report & baseline files (Total unique numeric points: {len(verified_numbers)}, Oracle reports: {len(oracle_reports)})")

    # Run self-tests if requested
    if args.test_wrong_claim:
        s1 = test_deliberate_wrong_number(verified_numbers)
        s2 = test_deliberate_broken_citation()
        s3 = test_deliberate_uncited_percentage_in_readme(verified_numbers)
        s4 = test_deliberate_oracle_citation_as_system_accuracy(oracle_reports)
        sys.exit(0 if (s1 and s2 and s3 and s4) else 1)

    all_violations = []

    # 1. Self-tests against deliberate wrong numbers, broken citations, uncited percentages, and oracle claims
    if not test_deliberate_wrong_number(verified_numbers):
        all_violations.append("Auditor self-test failed: could not detect fabricated 98.76% claim.")
    if not test_deliberate_broken_citation():
        all_violations.append("Auditor self-test failed: could not detect broken citation pointer.")
    if not test_deliberate_uncited_percentage_in_readme(verified_numbers):
        all_violations.append("Auditor self-test failed: could not detect uncited percentage in README.")
    if not test_deliberate_oracle_citation_as_system_accuracy(oracle_reports):
        all_violations.append("Auditor self-test failed: could not detect oracle citation as system accuracy.")

    # 2. Hardcoded confidence audit
    conf_violations = audit_hardcoded_confidences()
    if conf_violations:
        print(f"\n[FAIL] Found {len(conf_violations)} hardcoded confidence literals in backend/:")
        for v in conf_violations:
            print(f"  - {v}")
        all_violations.extend(conf_violations)
    else:
        print("[PASS] No hardcoded confidence literals in backend/ (all confidences dynamically derived).")

    # 3. /health honest provenance audit
    health_violations = audit_health_endpoint()
    if health_violations:
        print(f"\n[FAIL] Found {len(health_violations)} dishonest model declarations in /health:")
        for v in health_violations:
            print(f"  - {v}")
        all_violations.extend(health_violations)
    else:
        print("[PASS] /health declarations reflect actual runtime code and models.")

    # 4. Frontend copy audit
    ui_violations = audit_frontend_claims(verified_numbers)
    if ui_violations:
        print(f"\n[FAIL] Found {len(ui_violations)} unbacked claims in frontend/src/:")
        for v in ui_violations:
            print(f"  - {v}")
        all_violations.extend(ui_violations)
    else:
        print("[PASS] Frontend copy is free of fabricated claims and all percentages are report-backed.")

    # 5. README and active docs audit
    doc_violations = audit_readme_and_active_docs(verified_numbers, oracle_reports)
    if doc_violations:
        print(f"\n[FAIL] Found {len(doc_violations)} unbacked claims or broken citations in README.md / docs:")
        for v in doc_violations:
            print(f"  - {v}")
        all_violations.extend(doc_violations)
    else:
        print("[PASS] README.md and active docs/*.md are free of unverified claims and all file#key citations match.")

    print("\n" + "=" * 70)
    if all_violations:
        print(f"[SUMMARY] AUDIT FAILED with {len(all_violations)} violation(s).")
        print("=" * 70)
        sys.exit(1)
    else:
        print("[SUMMARY] ALL CLAIMS & PROVENANCE CHECKS PASSED.")
        print("=" * 70)
        sys.exit(0)

if __name__ == "__main__":
    main()
