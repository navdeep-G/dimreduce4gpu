from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__
from ._native import native_built, native_library_path, native_runnable


@dataclass
class DiagnoseInfo:
    version: str
    native_built: bool
    native_runnable: bool
    native_library_path: str | None


def _gather() -> DiagnoseInfo:
    path = native_library_path()
    return DiagnoseInfo(
        version=__version__,
        native_built=native_built(),
        native_runnable=native_runnable(),
        native_library_path=str(path) if path is not None else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dimreduce4gpu-diagnose",
        description="Print diagnostic information about dimreduce4gpu native/GPU availability.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    args = parser.parse_args(argv)

    info = _gather()
    payload = asdict(info)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"dimreduce4gpu {info.version}")
        print(f"native built:     {info.native_built}")
        print(f"native runnable:  {info.native_runnable}")
        print(f"native path:      {info.native_library_path or '<none>'}")

        if info.native_library_path and not Path(info.native_library_path).exists():
            print("note: native path was resolved but does not exist on disk")

        if info.native_built and not info.native_runnable:
            print(
                "hint: native library is present, but GPU execution is not available in this environment.\n"
                "      On Linux this usually means the NVIDIA driver (libcuda.so.1) is missing,\n"
                "      or no CUDA-capable GPU is present."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
