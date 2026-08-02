from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Mapping, Sequence

from jepa_lmc.checking.ctl import (
    AF,
    AG,
    AX,
    EF,
    EG,
    EU,
    EX,
    And,
    Atom,
    CTLModelChecker,
    Formula,
    Not,
    Or,
)
from jepa_lmc.checking.transition_system import (
    ActionT,
    ExplicitTransitionSystem,
    StateT,
)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#-]*$")
_VERDICT = re.compile(
    r"^-- specification\s+.*?\s+is\s+(true|false)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class CTLQuery(Generic[StateT]):
    """One CTL formula to evaluate at one specific state."""

    state: StateT
    formula: Formula
    name: str = ""


@dataclass(frozen=True)
class NuSMVRun:
    """Raw result returned by an independent NuSMV-compatible executable."""

    executable: str
    verdicts: tuple[bool, ...]
    output: str


@dataclass(frozen=True)
class DifferentialValidationReport(Generic[StateT]):
    """Comparison between this project and an independent CTL checker."""

    queries: tuple[CTLQuery[StateT], ...]
    internal_verdicts: tuple[bool, ...]
    external_verdicts: tuple[bool, ...]
    executable: str
    external_output: str

    @property
    def matches(self) -> tuple[bool, ...]:
        return tuple(
            internal == external
            for internal, external in zip(
                self.internal_verdicts,
                self.external_verdicts,
            )
        )

    @property
    def agreement(self) -> float:
        return sum(self.matches) / len(self.matches)

    @property
    def all_match(self) -> bool:
        return all(self.matches)


def formula_to_nusmv(
    formula: Formula,
    proposition_identifiers: Mapping[str, str] | None = None,
) -> str:
    """Translate the supported CTL fragment to NuSMV syntax."""

    def atom_identifier(name: str) -> str:
        if proposition_identifiers is not None:
            try:
                return proposition_identifiers[name]
            except KeyError as error:
                raise ValueError(
                    f"No NuSMV identifier was provided for proposition {name!r}."
                ) from error

        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(
                f"Proposition {name!r} is not a valid NuSMV identifier."
            )
        return name

    if isinstance(formula, Atom):
        return atom_identifier(formula.name)
    if isinstance(formula, Not):
        return f"!({formula_to_nusmv(formula.formula, proposition_identifiers)})"
    if isinstance(formula, And):
        left = formula_to_nusmv(formula.left, proposition_identifiers)
        right = formula_to_nusmv(formula.right, proposition_identifiers)
        return f"({left} & {right})"
    if isinstance(formula, Or):
        left = formula_to_nusmv(formula.left, proposition_identifiers)
        right = formula_to_nusmv(formula.right, proposition_identifiers)
        return f"({left} | {right})"
    if isinstance(formula, EX):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"EX ({child})"
    if isinstance(formula, AX):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"AX ({child})"
    if isinstance(formula, EF):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"EF ({child})"
    if isinstance(formula, AF):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"AF ({child})"
    if isinstance(formula, EG):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"EG ({child})"
    if isinstance(formula, AG):
        child = formula_to_nusmv(formula.formula, proposition_identifiers)
        return f"AG ({child})"
    if isinstance(formula, EU):
        condition = formula_to_nusmv(
            formula.condition,
            proposition_identifiers,
        )
        target = formula_to_nusmv(formula.target, proposition_identifiers)
        return f"E[({condition}) U ({target})]"
    raise TypeError(f"Unsupported CTL formula: {formula!r}")


def export_nusmv_queries(
    transition_system: ExplicitTransitionSystem[StateT, ActionT],
    queries: Sequence[CTLQuery[StateT]],
) -> str:
    """Export state-specific CTL queries as one self-contained NuSMV model.

    Every state is made initial.  A query at state ``s`` is encoded as
    ``state = s -> formula``.  NuSMV checks a SPEC in every initial state, so
    this guarded encoding yields exactly the truth value of the formula at
    the requested state while allowing all queries to run in one process.
    """
    query_tuple = tuple(queries)
    if not query_tuple:
        raise ValueError("At least one CTL query is required.")
    for query in query_tuple:
        transition_system.validate_state(query.state)

    states = tuple(
        sorted(
            transition_system.states,
            key=lambda state: (
                type(state).__module__,
                type(state).__qualname__,
                repr(state),
            ),
        )
    )
    state_identifiers = {
        state: f"s{index}" for index, state in enumerate(states)
    }
    propositions = sorted(
        {
            atom
            for query in query_tuple
            for atom in _formula_atoms(query.formula)
        }
    )
    proposition_identifiers = {
        proposition: f"ap{index}"
        for index, proposition in enumerate(propositions)
    }

    state_domain = ", ".join(state_identifiers[state] for state in states)
    lines = [
        "MODULE main",
        "VAR",
        f"  state : {{{state_domain}}};",
        "ASSIGN",
        f"  init(state) := {{{state_domain}}};",
        "  next(state) := case",
    ]
    for state in states:
        successors = sorted(
            transition_system.successors(state),
            key=lambda successor: state_identifiers[successor],
        )
        successor_set = ", ".join(
            state_identifiers[successor] for successor in successors
        )
        lines.append(
            f"    state = {state_identifiers[state]} : "
            f"{{{successor_set}}};"
        )
    lines.extend(("  esac;", "DEFINE"))

    for proposition in propositions:
        labelled_states = [
            state_identifiers[state]
            for state in states
            if proposition in transition_system.propositions(state)
        ]
        expression = (
            " | ".join(f"state = {state}" for state in labelled_states)
            if labelled_states
            else "FALSE"
        )
        lines.append(
            f"  {proposition_identifiers[proposition]} := {expression};"
        )

    for query in query_tuple:
        formula_text = formula_to_nusmv(
            query.formula,
            proposition_identifiers,
        )
        state_identifier = state_identifiers[query.state]
        lines.append(
            f"SPEC ((state = {state_identifier}) -> ({formula_text}))"
        )

    return "\n".join(lines) + "\n"


def find_nusmv_executable(explicit: str | Path | None = None) -> str | None:
    """Find nuXmv or NuSMV from an explicit path, environment, or PATH."""
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(os.fspath(explicit))
    candidates.extend(
        value
        for value in (
            os.environ.get("NUXMV_BINARY"),
            os.environ.get("NUSMV_BINARY"),
        )
        if value
    )
    candidates.extend(("nuXmv", "NuSMV", "nusmv"))

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    return None


def parse_nusmv_verdicts(output: str) -> tuple[bool, ...]:
    """Parse NuSMV's ordered ``SPEC ... is true/false`` verdicts."""
    verdicts = tuple(
        match.group(1).lower() == "true"
        for match in _VERDICT.finditer(output)
    )
    if not verdicts:
        raise ValueError("NuSMV output did not contain any CTL verdicts.")
    return verdicts


def run_nusmv_model(
    model_text: str,
    executable: str | Path | None = None,
) -> NuSMVRun:
    """Run an installed NuSMV-compatible binary on one generated model."""
    resolved = find_nusmv_executable(executable)
    if resolved is None:
        raise FileNotFoundError(
            "nuXmv/NuSMV was not found. Put it on PATH or set "
            "NUXMV_BINARY or NUSMV_BINARY to the executable path."
        )

    with tempfile.TemporaryDirectory(prefix="jepa_lmc_nusmv_") as directory:
        model_path = Path(directory) / "validation.smv"
        model_path.write_text(model_text, encoding="utf-8")
        completed = subprocess.run(
            [resolved, str(model_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"NuSMV exited with status {completed.returncode}.\n{output}"
        )
    return NuSMVRun(
        executable=resolved,
        verdicts=parse_nusmv_verdicts(output),
        output=output,
    )


def compare_queries_with_nusmv(
    transition_system: ExplicitTransitionSystem[StateT, ActionT],
    queries: Sequence[CTLQuery[StateT]],
    executable: str | Path | None = None,
) -> DifferentialValidationReport[StateT]:
    """Compare our CTL verdicts with an independent NuSMV execution."""
    query_tuple = tuple(queries)
    model_text = export_nusmv_queries(transition_system, query_tuple)
    internal_checker = CTLModelChecker(transition_system)
    internal_verdicts = tuple(
        internal_checker.holds(query.state, query.formula)
        for query in query_tuple
    )
    external_run = run_nusmv_model(model_text, executable)
    if len(external_run.verdicts) != len(query_tuple):
        raise RuntimeError(
            "NuSMV returned an unexpected number of verdicts: "
            f"expected {len(query_tuple)}, got {len(external_run.verdicts)}."
        )
    return DifferentialValidationReport(
        queries=query_tuple,
        internal_verdicts=internal_verdicts,
        external_verdicts=external_run.verdicts,
        executable=external_run.executable,
        external_output=external_run.output,
    )


def _formula_atoms(formula: Formula) -> frozenset[str]:
    if isinstance(formula, Atom):
        return frozenset((formula.name,))
    if isinstance(formula, (Not, EX, AX, EF, AF, EG, AG)):
        return _formula_atoms(formula.formula)
    if isinstance(formula, (And, Or)):
        return _formula_atoms(formula.left) | _formula_atoms(formula.right)
    if isinstance(formula, EU):
        return _formula_atoms(formula.condition) | _formula_atoms(formula.target)
    raise TypeError(f"Unsupported CTL formula: {formula!r}")
