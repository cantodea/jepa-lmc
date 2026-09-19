"""Audit the existing six CTL formulas and two CTL/LTL pairs on saved graphs."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import time
from pathlib import Path

from jepa_lmc.benchmarks.ctl_suite import default_ctl_suite
from jepa_lmc.benchmarks.ltl_suite import default_ltl_suite
from jepa_lmc.verification.ctl import CTLModelChecker
from jepa_lmc.verification.nuxmv import (
    _nusmv_model_preamble,
    find_nusmv_executable,
    formula_to_nusmv,
    ltl_formula_to_nusmv,
)

if __package__:
    from .evaluate_top1_ltl import digest, load_system, read_run
else:
    from evaluate_top1_ltl import digest, load_system, read_run

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("AG !danger", "G !danger"), ("AF goal", "F goal"))
SPEC_LINE = re.compile(r"^(SPEC|LTLSPEC) NAME (\w+) := (.+)$", re.MULTILINE)
VERDICT_LINE = re.compile(
    r"^-- specification\s+(.+?)\s+is\s+(true|false)\s*$", re.MULTILINE
)
CHECK_FIELDS = (
    "comparison",
    "seed",
    "map",
    "family",
    "graph",
    "state_id",
    "row",
    "column",
    "initial",
    "left_formula",
    "right_formula",
    "left_backend",
    "right_backend",
    "left_verdict",
    "right_verdict",
    "agreement",
)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_model(record: dict, graph: str):
    system = load_system(record, graph)
    ctl = default_ctl_suite()
    ltl = {p.name: p for p in default_ltl_suite()}
    # The six existing CTL formulas cover all three benchmark propositions.
    # Preserve any other saved labels too, without introducing any new queries.
    propositions = {"safe", "goal", "danger"}
    propositions.update(ap for s in system.states for ap in system.propositions(s))
    lines, state_ids, ap_ids = _nusmv_model_preamble(system, propositions)
    manifest = []
    for logic, properties, render in (
        ("ctl", ctl, formula_to_nusmv),
        ("ltl", tuple(ltl[name] for _, name in PAIRS), ltl_formula_to_nusmv),
    ):
        for state in range(len(record["states"])):
            for prop in properties:
                query_id = f"{logic}_{state}_{len(manifest)}"
                formula = render(prop.formula, ap_ids)
                expression = f"((state = {state_ids[state]}) -> ({formula}))"
                keyword = "SPEC" if logic == "ctl" else "LTLSPEC"
                lines.append(f"{keyword} NAME {query_id} := {expression}")
                manifest.append(
                    {
                        "id": query_id,
                        "logic": logic,
                        "state_id": state,
                        "formula": prop.name,
                        "expression": expression,
                    }
                )
    model = "\n".join(lines) + "\n"
    audit_encoding(system, model, state_ids, ap_ids)
    return system, model, manifest, state_ids, ap_ids


def audit_encoding(system, model: str, state_ids: dict, ap_ids: dict) -> None:
    """Independently read back the restricted SMV preamble into sets of edges/labels."""
    domain_match = re.search(r"state : \{([^}]+)\};", model)
    initial_match = re.search(r"init\(state\) := \{([^}]+)\};", model)
    if domain_match is None or initial_match is None:
        raise AssertionError("Missing state domain or initial encoding.")
    domain = {s.strip() for s in domain_match[1].split(",")}
    initial = {s.strip() for s in initial_match[1].split(",")}
    if len(state_ids) != len(set(state_ids.values())) or domain != set(
        state_ids.values()
    ):
        raise AssertionError("State identifiers are not a bijection.")
    if initial != domain:
        raise AssertionError(
            "State-specific queries must share the all-initial encoding."
        )
    if re.search(r"\b(FAIRNESS|JUSTICE|COMPASSION)\b", model):
        raise AssertionError("Fairness is outside this audit's semantics.")
    edges = re.findall(r"^\s*state = (s\d+) : \{([^}]+)\};$", model, re.MULTILINE)
    if len(edges) != len(system.states):
        raise AssertionError("Transition table is incomplete or duplicated.")
    parsed = {s: {t.strip() for t in targets.split(",")} for s, targets in edges}
    expected = {
        state_ids[s]: {state_ids[t] for t in system.successors(s)}
        for s in system.states
    }
    if parsed != expected:
        raise AssertionError("Encoded edges differ from the saved graph.")
    definitions = re.findall(r"^\s*(ap\d+) := (.+);$", model, re.MULTILINE)
    if len(definitions) != len(ap_ids):
        raise AssertionError("Atomic proposition definitions differ.")
    parsed_labels = {}
    for ap, expression in definitions:
        terms = expression.split(" | ")
        if expression == "FALSE":
            parsed_labels[ap] = set()
        elif all(re.fullmatch(r"state = s\d+", t) for t in terms):
            parsed_labels[ap] = {t.removeprefix("state = ") for t in terms}
        else:
            raise AssertionError("Unexpected proposition encoding.")
    expected_labels = {
        alias: {state_ids[s] for s in system.states if ap in system.propositions(s)}
        for ap, alias in ap_ids.items()
    }
    if parsed_labels != expected_labels:
        raise AssertionError("Encoded labels differ from the saved graph.")


def parse_verdicts(model: str, output: str) -> dict[str, bool]:
    # For this fixed suite, nuXmv's pretty-printer only changes whitespace and
    # parentheses. Match the actual state/formula expression, not output order.
    # Unknown, duplicate, transformed or missing expressions fail closed.
    def normalize(expression: str) -> str:
        return re.sub(r"[\s()]", "", expression)

    clauses = SPEC_LINE.findall(model)
    expected = {normalize(expression): name for _, name, expression in clauses}
    if not clauses or len(expected) != len(clauses):
        raise ValueError("Query expressions are absent or ambiguous.")
    results = {}
    for expression, value in VERDICT_LINE.findall(output):
        query_id = expected.get(normalize(expression))
        if query_id is None or query_id in results:
            raise ValueError(f"Unexpected or duplicate nuXmv formula: {expression}")
        results[query_id] = value == "true"
    if len(results) != len(clauses):
        raise ValueError(f"Expected {len(clauses)} verdicts, got {len(results)}.")
    return results


def agreement(rows: list[dict]) -> dict:
    matched = sum(r["agreement"] for r in rows)
    return {
        "comparisons": len(rows),
        "matched": matched,
        "agreement": matched / len(rows) if rows else None,
        "mismatches": len(rows) - matched,
        "left_true": sum(r["left_verdict"] for r in rows),
        "right_true": sum(r["right_verdict"] for r in rows),
    }


def run_audit(
    run_dir: Path, output_dir: Path, executable=None, timeout: float = 120
) -> dict:
    source, records = read_run(run_dir)
    resolved = find_nusmv_executable(executable)
    if resolved is None:
        raise FileNotFoundError(
            "nuXmv is required. Set NUXMV_BINARY or --executable PATH."
        )
    binary = Path(resolved).resolve()
    if timeout <= 0:
        raise ValueError("The per-model timeout must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    metadata = {(r["seed"], r["case"]): r for r in source["maps"]}
    for row in metadata.values():
        if "initial_ctl_disagreements" not in row:
            raise ValueError("Source report lacks the original initial CTL comparison.")
    inputs = {
        name: digest(run_dir / name) for name in ("report.json", "relations.jsonl")
    }
    binary_hash = digest(binary)
    sources = (
        "experiments/backend_sanity.py",
        "experiments/evaluate_top1_ltl.py",
        "src/jepa_lmc/verification/ctl.py",
        "src/jepa_lmc/verification/ltl.py",
        "src/jepa_lmc/verification/nuxmv.py",
        "src/jepa_lmc/verification/transition_system.py",
        "src/jepa_lmc/benchmarks/ctl_suite.py",
        "src/jepa_lmc/benchmarks/ltl_suite.py",
    )
    source_hashes = {p: digest(ROOT / p) for p in sources}
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = output_dir / "backend"
    backend_dir.mkdir()
    checks, initials, initial_formulas, baseline_changes = [], [], [], []
    banners = set()
    started = time.perf_counter()
    for index, record in enumerate(records):
        seed, case = record["seed"], record["case"]
        meta = metadata[seed, case]
        values = {}
        for graph in ("real", "top1"):
            system, model, manifest, state_ids, ap_ids = build_model(record, graph)
            stem = backend_dir / f"{index:03d}_{graph}"
            model_path = stem.with_suffix(".smv")
            model_path.write_text(model, encoding="utf-8")
            write_json(
                stem.with_suffix(".json"),
                {
                    "seed": seed,
                    "map": case,
                    "graph": graph,
                    "saved_initial_state_id": record["initial"],
                    "state_catalogue": record["states"],
                    "state_identifiers": state_ids,
                    "proposition_identifiers": ap_ids,
                    "queries": manifest,
                },
            )
            # Exact finite-state batch checking. Explicit -s disables startup files;
            # no BMC horizon, fairness declarations or additional formulas are used.
            completed = subprocess.run(
                [str(binary), "-s", str(model_path.resolve())],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            output = completed.stdout + completed.stderr
            stem.with_suffix(".txt").write_text(output, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"nuXmv failed for {seed}/{case}/{graph}: {output}")
            banner = next(
                (s for s in output.splitlines() if s.startswith("*** This is nuXmv ")),
                None,
            )
            if banner is None:
                raise RuntimeError(
                    "The requested backend must identify itself as nuXmv."
                )
            banners.add(banner)
            external = parse_verdicts(model, output)
            checker = CTLModelChecker(system)
            native = {
                (state, prop.name): checker.holds(state, prop.formula)
                for state in range(len(record["states"]))
                for prop in default_ctl_suite()
            }
            ctl_values = {
                (q["state_id"], q["formula"]): external[q["id"]]
                for q in manifest
                if q["logic"] == "ctl"
            }
            ltl_values = {
                (q["state_id"], q["formula"]): external[q["id"]]
                for q in manifest
                if q["logic"] == "ltl"
            }
            values[graph] = (native, ctl_values)
            for state, (row, column) in enumerate(record["states"]):
                context = {
                    "seed": seed,
                    "map": case,
                    "family": meta["family"],
                    "graph": graph,
                    "state_id": state,
                    "row": row,
                    "column": column,
                    "initial": state == record["initial"],
                }
                comparisons = [
                    (
                        "internal_ctl_vs_nuxmv_ctl",
                        p.name,
                        p.name,
                        "CTLModelChecker",
                        "nuXmv CTL",
                        native[state, p.name],
                        ctl_values[state, p.name],
                    )
                    for p in default_ctl_suite()
                ] + [
                    (
                        "nuxmv_ctl_vs_nuxmv_ltl",
                        ctl,
                        ltl,
                        "nuXmv CTL",
                        "nuXmv LTL",
                        ctl_values[state, ctl],
                        ltl_values[state, ltl],
                    )
                    for ctl, ltl in PAIRS
                ]
                for kind, left, right, lb, rb, lv, rv in comparisons:
                    checks.append(
                        {
                            **context,
                            "comparison": kind,
                            "left_formula": left,
                            "right_formula": right,
                            "left_backend": lb,
                            "right_backend": rb,
                            "left_verdict": lv,
                            "right_verdict": rv,
                            "agreement": lv == rv,
                        }
                    )
        start = record["initial"]
        old_differences = set(meta["initial_ctl_disagreements"])
        native_differences, external_differences = [], []
        for prop in default_ctl_suite():
            ir, er = (v[start, prop.name] for v in values["real"])
            ih, eh = (v[start, prop.name] for v in values["top1"])
            if ir != ih:
                native_differences.append(prop.name)
            if er != eh:
                external_differences.append(prop.name)
            initial_formulas.append(
                {
                    "seed": seed,
                    "map": case,
                    "state_id": start,
                    "row": record["states"][start][0],
                    "column": record["states"][start][1],
                    "formula": prop.name,
                    "internal_real": ir,
                    "internal_top1": ih,
                    "nuxmv_real": er,
                    "nuxmv_top1": eh,
                    "saved_models_agree": prop.name not in old_differences,
                    "nuxmv_models_agree": er == eh,
                }
            )
            if (prop.name not in old_differences) != (er == eh):
                baseline_changes.append(
                    {
                        "comparison": "saved_initial_vs_nuxmv_initial",
                        "seed": seed,
                        "map": case,
                        "family": meta["family"],
                        "graph": "real_vs_top1",
                        "state_id": start,
                        "row": record["states"][start][0],
                        "column": record["states"][start][1],
                        "initial": True,
                        "left_formula": prop.name,
                        "right_formula": prop.name,
                        "left_backend": "saved real/Top1 agreement",
                        "right_backend": "nuXmv CTL real/Top1 agreement",
                        "left_verdict": prop.name not in old_differences,
                        "right_verdict": er == eh,
                        "agreement": False,
                    }
                )
        initials.append(
            {
                "seed": seed,
                "map": case,
                "family": meta["family"],
                "state": record["states"][start],
                "saved_all_six": not old_differences,
                "internal_all_six": not native_differences,
                "nuxmv_all_six": not external_differences,
                "saved_disagreements": sorted(old_differences),
                "internal_disagreements": native_differences,
                "nuxmv_disagreements": external_differences,
            }
        )
        print(
            f"[{index + 1}/{len(records)}] {seed} {case}: checked both graphs",
            flush=True,
        )
    if inputs != {name: digest(run_dir / name) for name in inputs}:
        raise AssertionError("Saved input files changed during the audit.")
    if binary_hash != digest(binary) or len(banners) != 1:
        raise AssertionError("Backend identity changed during the audit.")
    if source_hashes != {p: digest(ROOT / p) for p in sources}:
        raise AssertionError("Checker source changed during the audit.")
    ctl_checks = [r for r in checks if r["comparison"] == "internal_ctl_vs_nuxmv_ctl"]
    paired_checks = [r for r in checks if r["comparison"] == "nuxmv_ctl_vs_nuxmv_ltl"]
    mismatches = [r for r in checks if not r["agreement"]] + baseline_changes
    write_csv(output_dir / "checks.csv", checks, CHECK_FIELDS)
    write_csv(output_dir / "mismatches.csv", mismatches, CHECK_FIELDS)
    write_csv(
        output_dir / "initial_formulas.csv", initial_formulas, list(initial_formulas[0])
    )
    report = {
        "status": "passed" if not mismatches else "mismatch",
        "saved_graph_cases": len(records),
        "graphs_checked": 2 * len(records),
        "graph_states_checked": sum(len(r["states"]) for r in records) * 2,
        "source_run_dir": str(run_dir.resolve()),
        "source_run_revision": source["source_revision"],
        "source_input_sha256": inputs,
        "checker_source_sha256": source_hashes,
        "checker_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": {
            "executable": str(binary),
            "sha256": binary_hash,
            "banner": sorted(banners)[0],
            "arguments": ["-s"],
            "mode": "exact finite-state batch checking; no BMC bound",
        },
        "encoding": {
            "shared_ctl_ltl_file_and_process": True,
            "fairness": False,
            "actions": "ignored",
            "labels": "all saved propositions",
            "initial": "all states, with the same state-specific implication guard",
            "designated_initial": "saved initial index used for the all-six score",
            "independent_graph_label_encoding_audits": 2 * len(records),
        },
        "input_backend_and_sources_unchanged": True,
        "external_backend_calls": 2 * len(records),
        "external_ctl_verdicts": len(ctl_checks),
        "external_ltl_verdicts": len(paired_checks),
        "internal_ctl_vs_nuxmv_ctl": agreement(ctl_checks),
        "internal_ctl_vs_nuxmv_ctl_by_property": {
            p.name: agreement([r for r in ctl_checks if r["left_formula"] == p.name])
            for p in default_ctl_suite()
        },
        "paired_formulas": {
            f"{ctl} <=> {ltl}": agreement(
                [r for r in paired_checks if r["left_formula"] == ctl]
            )
            for ctl, ltl in PAIRS
        },
        "initial_all_six": {
            "maps": len(initials),
            "saved": sum(r["saved_all_six"] for r in initials),
            "internal": sum(r["internal_all_six"] for r in initials),
            "nuxmv": sum(r["nuxmv_all_six"] for r in initials),
            "changed_maps": sum(
                r["saved_all_six"] != r["nuxmv_all_six"] for r in initials
            ),
            "changed_formula_agreements": len(baseline_changes),
        },
        "by_seed": {
            str(seed): {
                "internal_ctl_vs_nuxmv_ctl": agreement(
                    [r for r in ctl_checks if r["seed"] == seed]
                ),
                "paired_formulas": agreement(
                    [r for r in paired_checks if r["seed"] == seed]
                ),
                "initial_all_six": sum(
                    r["nuxmv_all_six"] for r in initials if r["seed"] == seed
                ),
            }
            for seed in sorted({r["seed"] for r in initials})
        },
        "by_graph": {
            g: agreement([r for r in ctl_checks if r["graph"] == g])
            for g in ("real", "top1")
        },
        "mismatch_count": len(mismatches),
        "initial_maps": initials,
        "elapsed_seconds": time.perf_counter() - started,
        "exports": {
            p.relative_to(output_dir).as_posix(): digest(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "internal_ctl_vs_nuxmv_ctl",
                    "paired_formulas",
                    "initial_all_six",
                    "mismatch_count",
                )
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument(
        "--timeout", type=float, default=120, help="Seconds per graph/backend call."
    )
    args = parser.parse_args()
    report = run_audit(args.run_dir, args.output_dir, args.executable, args.timeout)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
