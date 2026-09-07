"""Deterministic pre-execution misuse scan for agent-generated code.

This is NOT a sandbox and does not close the accepted risk named in
src/services/code_execution.py's docstring and ADR-001/ADR-003 (filesystem
and network access from the generated script remain technically
unenforced). It sits in front of that still-open boundary as a second,
much cheaper line of defence: a denylist over the script's AST, run before
the script is ever written to disk or executed.

Two realistic threats motivate it, given this project's own architecture
(no cloud call, a local 14b model, and dataset content - column names, cell
values - embedded verbatim into the modeller's prompt by src/profiling.py):
(1) the local model going wrong or getting confused into writing dangerous
code on its own, and (2) prompt injection carried in an uploaded CSV
steering the model into writing something harmful. Both are things a
denylist over recognisable dangerous calls can plausibly catch, because
neither depends on the code being deliberately written to evade detection.

What this cannot do, stated plainly rather than left implicit: it cannot
catch a script that is deliberately written to evade it. Building the
string "os" via concatenation or chr() codes and reaching it with
getattr(__builtins__, ...), monkeypatching a module before importing it,
using a benign-looking third-party import that itself shells out
internally, or routing a call through a local variable ("f = eval; f(x)")
all defeat this scan completely. It is a denylist over syntax, not a
sandbox, not a taint tracker, and not a proof of safety - a clean scan
result means "no recognised dangerous pattern was written in an obvious
way", nothing stronger.

Detection is AST-based rather than substring/regex matching on the source
text, deliberately not reusing src/checks.py's approach: that module's
checks are heuristics over generated training code where a false positive
just adds noise to a critic's context, but here a false positive silently
blocks a legitimate run before it ever executes, so matching needs to
understand what a name actually refers to (an import, an alias, a call),
not just that a substring appears somewhere in the file.

Import aliasing (`import X as Y`, `from X import Y as Z`) is resolved
against a single flat map built by walking the whole module once - not
scoped per function, not full data-flow analysis. A rename inside a
function body is still caught; a rename that only takes effect via
runtime control flow (an alias assigned conditionally, or a call routed
through a variable) is not. This matches the honesty note above: aliasing
resolution here defeats a lazy rename, not a deliberate one.

Absolute-path detection only looks at string-literal arguments, by design
(see scan_code's docstring) - a variable such as `train_path` cannot be
checked without running the code, and this module never runs anything.

An absolute literal under this project's own data/ directory is exempted
from that check, not flagged: src/agent.py hands the modeller an absolute
train_path (dataset.TRAIN_PATH) as part of its own prompt, and a correctly-
behaving script is expected to echo that literal straight into
pd.read_csv(...) - src/evaluation.py's fixture paths are absolute under
data/processed/eval_fixtures/ for the same structural reason. Flagging
those would block the large majority of entirely legitimate runs, not catch
misuse. A '..' traversal segment is still flagged unconditionally, including
inside data/, since it can walk back out regardless of where it starts.
"""

import ast
import os
import re
from dataclasses import dataclass

from src.config import DATA_DIR

# Categories, as literal strings, are part of this module's frozen public
# interface (a parallel red-team harness matches on them directly) - do not
# rename or otherwise vary these even though "deserialization"/
# "optimization"-style spellings elsewhere in this file's prose stay British.
CATEGORY_NETWORK_ACCESS = "network_access"
CATEGORY_PROCESS_EXECUTION = "process_execution"
CATEGORY_DYNAMIC_CODE_EXECUTION = "dynamic_code_execution"
CATEGORY_UNSAFE_DESERIALIZATION = "unsafe_deserialization"
CATEGORY_DESTRUCTIVE_FILESYSTEM_OP = "destructive_filesystem_op"
CATEGORY_ABSOLUTE_PATH_ACCESS = "absolute_path_access"


@dataclass(frozen=True)
class GuardrailFinding:
    category: str
    detail: str


# Dotted import names (exactly as they would appear after "import " or after
# "from X import " when combined as "X.Y") that are dangerous purely by
# virtue of being imported at all. Exact-match only, not a prefix check -
# "urllib" and "urllib.request" are both listed separately here precisely so
# "import urllib.request" matches only the more specific entry and does not
# also fire the bare "urllib" one.
_IMPORT_DENYLIST: dict[str, str] = {
    "socket": CATEGORY_NETWORK_ACCESS,
    "requests": CATEGORY_NETWORK_ACCESS,
    "urllib": CATEGORY_NETWORK_ACCESS,
    "urllib.request": CATEGORY_NETWORK_ACCESS,
    "urllib3": CATEGORY_NETWORK_ACCESS,
    "http.client": CATEGORY_NETWORK_ACCESS,
    "ftplib": CATEGORY_NETWORK_ACCESS,
    "smtplib": CATEGORY_NETWORK_ACCESS,
    "paramiko": CATEGORY_NETWORK_ACCESS,
    "aiohttp": CATEGORY_NETWORK_ACCESS,
    "telnetlib": CATEGORY_NETWORK_ACCESS,
    "pty": CATEGORY_PROCESS_EXECUTION,
    "ctypes": CATEGORY_DYNAMIC_CODE_EXECUTION,
}

# The real os.exec*/os.spawn* variants, spelled out rather than matched by
# prefix - a prefix test like resolved.startswith("os.exec") would also fire
# on an unrelated method that merely starts with "exec" (e.g. a hypothetical
# os.execute(...)), silently blocking a legitimate run over a name that was
# never actually one of Python's process-replacement functions.
_OS_EXEC_SPAWN_NAMES = {
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.posix_spawn", "os.posix_spawnp",
}

# Fully-resolved dotted call names (module.function, or a bare builtin name
# with no dot) mapped to the category they are evidence for. Keeping the dot
# in every non-builtin entry is what stops a builtin match (e.g. "compile")
# from ever firing against an unrelated attribute call that merely happens
# to share the same final segment (e.g. "zipfile.compile" - not a real
# example, but the shape "obj.open(...)" vs bare "open(...)" is exactly the
# kind of collision this dot-inclusion avoids without extra special-casing).
_CALL_DENYLIST: dict[str, str] = {
    "os.system": CATEGORY_PROCESS_EXECUTION,
    "os.popen": CATEGORY_PROCESS_EXECUTION,
    "subprocess.run": CATEGORY_PROCESS_EXECUTION,
    "subprocess.Popen": CATEGORY_PROCESS_EXECUTION,
    "subprocess.call": CATEGORY_PROCESS_EXECUTION,
    "subprocess.check_call": CATEGORY_PROCESS_EXECUTION,
    "subprocess.check_output": CATEGORY_PROCESS_EXECUTION,
    "eval": CATEGORY_DYNAMIC_CODE_EXECUTION,
    "exec": CATEGORY_DYNAMIC_CODE_EXECUTION,
    "compile": CATEGORY_DYNAMIC_CODE_EXECUTION,
    "__import__": CATEGORY_DYNAMIC_CODE_EXECUTION,
    "pickle.load": CATEGORY_UNSAFE_DESERIALIZATION,
    "pickle.loads": CATEGORY_UNSAFE_DESERIALIZATION,
    "joblib.load": CATEGORY_UNSAFE_DESERIALIZATION,
    "pandas.read_pickle": CATEGORY_UNSAFE_DESERIALIZATION,
    "shutil.rmtree": CATEGORY_DESTRUCTIVE_FILESYSTEM_OP,
    "os.remove": CATEGORY_DESTRUCTIVE_FILESYSTEM_OP,
    "os.unlink": CATEGORY_DESTRUCTIVE_FILESYSTEM_OP,
    "os.rmdir": CATEGORY_DESTRUCTIVE_FILESYSTEM_OP,
    "os.removedirs": CATEGORY_DESTRUCTIVE_FILESYSTEM_OP,
}

# Calls whose first positional argument gets the absolute-path/traversal
# check below, once resolved to one of these dotted names.
_FILE_ACCESS_CALLS = {
    "open",
    "pandas.read_csv",
    "pandas.read_excel",
    "pandas.read_json",
    "pandas.read_parquet",
    "pathlib.Path",
}

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")

# Resolved once against this project's own known directory, not against
# attacker-controlled content - _is_dangerous_path below only ever does a
# plain string comparison against this, never a filesystem call on the
# generated code's own path literal.
_DATA_DIR_PREFIX = str(DATA_DIR.resolve())


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
    """Maps a local name to the real dotted name it refers to, for every
    `import X as Y` and `from X import Y as Z` in the module (including
    inside function/class bodies - one flat map, not scoped per function;
    see this module's docstring on why that is an accepted simplification).

    A plain `import os` (no rename) deliberately gets no entry: code then
    refers to it as `os.system(...)`, which the attribute-resolution logic
    in _resolve_call_name already reads correctly with no map lookup at
    all. Only an actual rename needs tracking here.
    """
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    alias_map[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue  # cannot resolve a star-import statically - a real gap, not fixed here
                full_name = f"{module}.{alias.name}" if module else alias.name
                local_name = alias.asname or alias.name
                alias_map[local_name] = full_name
    return alias_map


def _resolve_call_name(node: ast.Call, alias_map: dict[str, str]) -> str | None:
    """Resolves a Call node's target to the dotted name it really refers to,
    substituting any tracked alias. Returns None for a call shape this
    module does not attempt to reason about (anything other than a bare
    name or a one-level `base.attr` attribute access on a bare name) -
    deeper chains, calls returned by another call, subscripts, and so on
    are exactly the kind of dynamic shape the module docstring already
    concedes it cannot see through.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return alias_map.get(func.id, func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = alias_map.get(func.value.id, func.value.id)
        return f"{base}.{func.attr}"
    return None


def _has_allow_pickle_true(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "allow_pickle" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
            return True
    return False


def _is_dangerous_path(path: str) -> bool:
    """A '..' traversal segment (flagged regardless of where it starts, since
    it can walk back out of data/ from inside it), or an absolute path that
    resolves outside this project's own data/ directory. Splits on both
    slash styles rather than checking `".." in path` directly - the
    substring check would misfire on an entirely ordinary filename like
    "model..txt", which contains the two characters ".." without containing
    a traversal segment at all."""
    if ".." in re.split(r"[\\/]", path):
        return True
    if path.startswith("~"):
        # A literal home-directory reference (~/.ssh/id_rsa, ~/.aws/credentials)
        # is never a legitimate training-script path and is always flagged,
        # unlike an absolute path - there is no equivalent "expected" case to
        # exempt it against.
        return True
    if not (path.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(path)):
        return False
    normalised = os.path.normpath(path)
    return not (normalised == _DATA_DIR_PREFIX or normalised.startswith(_DATA_DIR_PREFIX + os.sep))


def scan_code(code: str) -> list[GuardrailFinding]:
    """Statically scans `code` for recognisable dangerous patterns before it
    is ever written to disk or executed. Pure: no exec/eval of `code`, no
    file I/O, no network - only ast.parse and tree traversal.

    Returns [] if `code` fails to parse. A script that doesn't even parse is
    not this function's concern - the sandbox's own subprocess call already
    surfaces that as an ordinary failed run with the real SyntaxError
    traceback in stderr, and that existing behaviour must not change here.
    ValueError is caught alongside SyntaxError because a source string
    containing a null byte raises ValueError from ast.parse, not
    SyntaxError - and an uploaded CSV cell is a plausible source of one.

    Findings are returned sorted by line number. ast.walk does not visit
    nodes in source order, and this function's output ends up verbatim in a
    stored ExecutionResult.stderr (and from there, potentially a saved
    RunReport) - non-deterministic ordering would make otherwise-identical
    runs produce different-looking failure records.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []

    alias_map = _build_alias_map(tree)
    findings: list[tuple[int, GuardrailFinding]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                category = _IMPORT_DENYLIST.get(alias.name)
                if category is not None:
                    findings.append(
                        (node.lineno, GuardrailFinding(category, f"import of '{alias.name}' at line {node.lineno}"))
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                full_name = f"{module}.{alias.name}" if module else alias.name
                # Check the more specific combined name first, e.g. "from
                # urllib import request" should report "urllib.request", not
                # the less precise "urllib".
                category = _IMPORT_DENYLIST.get(full_name) or _IMPORT_DENYLIST.get(module)
                reported_name = full_name if full_name in _IMPORT_DENYLIST else module
                if category is not None:
                    findings.append(
                        (node.lineno, GuardrailFinding(category, f"import of '{reported_name}' at line {node.lineno}"))
                    )
        elif isinstance(node, ast.Call):
            resolved = _resolve_call_name(node, alias_map)
            if resolved is None:
                continue
            if resolved in _OS_EXEC_SPAWN_NAMES:
                findings.append(
                    (node.lineno, GuardrailFinding(CATEGORY_PROCESS_EXECUTION, f"{resolved} call at line {node.lineno}"))
                )
            elif resolved in _CALL_DENYLIST:
                findings.append(
                    (node.lineno, GuardrailFinding(_CALL_DENYLIST[resolved], f"{resolved} call at line {node.lineno}"))
                )
            elif resolved == "numpy.load" and _has_allow_pickle_true(node):
                findings.append(
                    (
                        node.lineno,
                        GuardrailFinding(
                            CATEGORY_UNSAFE_DESERIALIZATION,
                            f"numpy.load call with allow_pickle=True at line {node.lineno}",
                        ),
                    )
                )
            elif resolved in _FILE_ACCESS_CALLS and node.args:
                first_arg = node.args[0]
                if (
                    isinstance(first_arg, ast.Constant)
                    and isinstance(first_arg.value, str)
                    and _is_dangerous_path(first_arg.value)
                ):
                    findings.append(
                        (
                            node.lineno,
                            GuardrailFinding(
                                CATEGORY_ABSOLUTE_PATH_ACCESS,
                                f"{resolved} call with path '{first_arg.value}' at line {node.lineno}",
                            ),
                        )
                    )

    findings.sort(key=lambda pair: pair[0])
    return [finding for _, finding in findings]
