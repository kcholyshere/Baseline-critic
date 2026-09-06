"""Local MCP server exposing the real, installed API surface for the
libraries the baseline agent's generated code depends on (pandas,
scikit-learn, LightGBM).

Grounds code generation in this project's actual installed versions
instead of a model's training data, which can be stale relative to a
library's current API (see the early_stopping_rounds failure recorded in
references/local-model-benchmarks.md). No network call and no third-party
account: the signature and docstring come from `inspect` against the
packages already installed in this project's own .venv, so they are
guaranteed to match exactly what src/services/code_execution.py will
actually run against.

Run as: python -m src.services.docs_mcp_server (stdio transport).
"""

import importlib
import inspect

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("baseline-critic-docs")

_ALLOWED_MODULES = {
    "pandas": "pandas",
    "sklearn": "sklearn",
    "lightgbm": "lightgbm",
}


@mcp.tool()
def get_api_signature(library: str, symbol: str) -> str:
    """Returns the real, currently installed signature and docstring for a
    dotted symbol path in one of pandas, sklearn, or lightgbm - e.g.
    library="lightgbm", symbol="train". Call this before using any
    function whose exact current keyword arguments you are not certain
    of: library APIs change between versions, and your training data may
    be stale relative to what is actually installed here.

    Args:
        library: One of "pandas", "sklearn", "lightgbm".
        symbol: A dotted path within that library, e.g. "train" or
            "Booster.save_model".
    """
    if library not in _ALLOWED_MODULES:
        return f"Unknown library '{library}'. Allowed: {', '.join(_ALLOWED_MODULES)}."

    module = importlib.import_module(_ALLOWED_MODULES[library])
    obj = module
    for part in symbol.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return f"'{symbol}' not found in {library} {module.__version__}."

    owning_module = getattr(obj, "__module__", "") or ""
    if not owning_module.startswith(_ALLOWED_MODULES[library]):
        return f"'{symbol}' not found in {library} {module.__version__}."

    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = "(signature unavailable)"
    doc = inspect.getdoc(obj) or ""
    return f"{library} {module.__version__}\n{symbol}{signature}\n\n{doc[:1500]}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
