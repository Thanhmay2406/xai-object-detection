from __future__ import annotations

import sys
from collections.abc import Sequence

from rcnn_odamTrain import train


def run_method(method_name: str, default_args: Sequence[str], argv: Sequence[str] | None = None) -> None:
    """Run the shared training engine with method-specific default flags.

    User-provided CLI flags are appended after defaults, so argparse's normal
    last-value-wins behavior lets an experiment override any preset.
    """

    extra_args = list(sys.argv[1:] if argv is None else argv)
    if "-h" not in extra_args and "--help" not in extra_args:
        print(f"method_entrypoint={method_name}", flush=True)
    train.main([*default_args, *extra_args])
