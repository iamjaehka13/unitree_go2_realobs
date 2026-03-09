#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    # .../<repo-root>/unitree_go2_realobs/scripts/real/prune_sdk2_for_go2.py
    return Path(__file__).resolve().parents[3]


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.1f}{units[idx]}"


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _write_guarded_example_cmake(example_cmake: Path) -> None:
    content = """# Go2-focused, presence-guarded example list.
set(UNITREE_SDK2_EXAMPLES
  helloworld
  wireless_controller
  jsonize
  state_machine
  go2
  b2
  h1
  g1
  go2w
  b2w
  a2
)

foreach(example_name IN LISTS UNITREE_SDK2_EXAMPLES)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/CMakeLists.txt")
    add_subdirectory(${example_name})
  endif()
endforeach()
"""
    example_cmake.write_text(content)


@dataclass
class PruneTarget:
    rel_path: str
    reason: str


def _default_targets() -> list[PruneTarget]:
    return [
        PruneTarget(".github", "CI metadata not used by local runs"),
        PruneTarget(".devcontainer", "Devcontainer files not needed here"),
        PruneTarget("example/a2", "Non-Go2 example"),
        PruneTarget("example/b2", "Non-Go2 example"),
        PruneTarget("example/b2w", "Non-Go2 example"),
        PruneTarget("example/g1", "Non-Go2 example"),
        PruneTarget("example/h1", "Non-Go2 example"),
        PruneTarget("example/go2w", "Non-Go2 example"),
        PruneTarget("example/helloworld", "Generic sample not required"),
        PruneTarget("example/jsonize", "Generic sample not required"),
        PruneTarget("example/state_machine", "Generic sample not required"),
        PruneTarget("example/wireless_controller", "Generic sample not required"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune Unitree SDK2 tree for Go2-only usage.")
    parser.add_argument(
        "--sdk-root",
        type=str,
        default=None,
        help="SDK root path (default: <repo>/third_party/unitree_sdk2)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove files/directories. Default is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly select dry-run mode (same as default).",
    )
    parser.add_argument(
        "--drop-go2-example",
        action="store_true",
        help="Also remove example/go2 (not recommended unless you already integrated your own bridge).",
    )
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run cannot be used together.")

    apply_mode = bool(args.apply)

    sdk_root = (
        Path(args.sdk_root).expanduser().resolve()
        if args.sdk_root
        else (_repo_root() / "third_party" / "unitree_sdk2")
    )
    if not sdk_root.exists() or not sdk_root.is_dir():
        print(f"[FAIL] SDK root not found: {sdk_root}")
        return 1

    targets = _default_targets()
    if args.drop_go2_example:
        targets.append(PruneTarget("example/go2", "Requested by --drop-go2-example"))

    print("=== SDK2 Prune Plan (Go2-only) ===")
    print(f"SDK root : {sdk_root}")
    print(f"Mode     : {'APPLY' if apply_mode else 'DRY-RUN'}")
    print("")

    existing: list[tuple[Path, str]] = []
    reclaim_bytes = 0
    for t in targets:
        p = sdk_root / t.rel_path
        if not p.exists():
            continue
        existing.append((p, t.reason))
        reclaim_bytes += _path_size_bytes(p)

    if len(existing) == 0:
        print("[INFO] Nothing to prune for current SDK tree.")
    else:
        print("Targets:")
        for p, reason in existing:
            print(f"  - {p.relative_to(sdk_root)} ({reason})")
        print(f"Estimated reclaim: {_human_size(reclaim_bytes)}")

    example_cmake = sdk_root / "example" / "CMakeLists.txt"
    if example_cmake.exists():
        print(f"CMake patch : {example_cmake.relative_to(sdk_root)} -> guarded add_subdirectory()")
    else:
        print("[WARN] example/CMakeLists.txt not found; skipping CMake patch.")

    if not apply_mode:
        print("")
        print("Dry-run complete. Re-run with --apply to execute.")
        return 0

    removed_count = 0
    for p, _reason in existing:
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p)
        removed_count += 1

    if example_cmake.exists():
        _write_guarded_example_cmake(example_cmake)

    print("")
    print(f"[DONE] Removed entries: {removed_count}")
    print(f"[DONE] Updated guarded CMake file: {example_cmake}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
