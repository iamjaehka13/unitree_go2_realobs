#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../<repo-root>/unitree_go2_realobs/scripts/real/check_sdk_setup.py
    # -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _print_expected_layout(sdk_root: Path) -> None:
    print("Expected SDK path:")
    print(f"  {sdk_root}")
    print("")
    print("Expected layout (minimum):")
    print("  - CMakeLists.txt")
    print("  - include/unitree/")
    print("  - thirdparty/include/")
    print("  - thirdparty/lib/")
    print("  - lib/<arch>/libunitree_sdk2.a")
    print("  - example/go2/")
    print("")


def _list_top_level(path: Path) -> list[str]:
    items = sorted([p.name for p in path.iterdir()])
    return items[:20]


def _guess_host_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def _nested_sdk_hint(sdk_root: Path) -> str:
    items = list(sdk_root.iterdir())
    if len(items) != 1 or not items[0].is_dir():
        return ""
    nested = items[0]
    nested_required = [
        nested / "CMakeLists.txt",
        nested / "include",
        nested / "lib",
    ]
    if all(p.exists() for p in nested_required):
        return f"[HINT] SDK appears nested. Try: --sdk-root {nested}"
    return ""


def validate_sdk_root(sdk_root: Path, target_arch: str) -> tuple[bool, list[str]]:
    messages: list[str] = []
    failures: list[str] = []

    if not sdk_root.exists():
        return False, [f"[FAIL] SDK path does not exist: {sdk_root}"]
    if not sdk_root.is_dir():
        return False, [f"[FAIL] SDK path is not a directory: {sdk_root}"]

    items = list(sdk_root.iterdir())
    if len(items) == 0:
        return False, [f"[FAIL] SDK directory is empty: {sdk_root}"]

    required_entries: list[tuple[str, str]] = [
        ("CMakeLists.txt", "file"),
        ("README.md", "file"),
        ("include/unitree", "dir"),
        ("thirdparty/include", "dir"),
        ("thirdparty/lib", "dir"),
        ("example/go2", "dir"),
    ]
    for rel, kind in required_entries:
        p = sdk_root / rel
        exists = p.is_file() if kind == "file" else p.is_dir()
        if exists:
            messages.append(f"[OK] {rel}")
        else:
            failures.append(f"[FAIL] Missing {kind}: {rel}")

    lib_candidates = {
        "x86_64": sdk_root / "lib" / "x86_64" / "libunitree_sdk2.a",
        "aarch64": sdk_root / "lib" / "aarch64" / "libunitree_sdk2.a",
    }
    available_arches = [arch for arch, path in lib_candidates.items() if path.exists()]
    if len(available_arches) == 0:
        failures.append("[FAIL] Missing static SDK library: lib/<arch>/libunitree_sdk2.a")
    else:
        messages.append(f"[OK] Available library arches: {', '.join(sorted(available_arches))}")

    host_arch = _guess_host_arch()
    if target_arch == "auto":
        check_arch = host_arch
    elif target_arch == "any":
        check_arch = ""
    else:
        check_arch = target_arch

    messages.append(f"[INFO] Host arch guess: {host_arch}")
    if check_arch != "":
        messages.append(f"[INFO] Target arch check: {check_arch}")
        if check_arch in lib_candidates and not lib_candidates[check_arch].exists():
            failures.append(
                f"[FAIL] Missing lib for target arch '{check_arch}': {lib_candidates[check_arch]}"
            )
    else:
        messages.append("[INFO] Target arch check skipped (--target-arch any).")

    top_level = _list_top_level(sdk_root)
    messages.append(f"[INFO] Top-level entries: {', '.join(top_level)}")
    hint = _nested_sdk_hint(sdk_root)
    if hint != "":
        messages.append(hint)

    if len(failures) > 0:
        messages.extend(failures)
        return False, messages

    return True, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Unitree SDK2 placement for this repo.")
    parser.add_argument(
        "--sdk-root",
        type=str,
        default=None,
        help="Override SDK root path. Default: <repo>/third_party/unitree_sdk2",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default="auto",
        choices=["auto", "x86_64", "aarch64", "any"],
        help="Required SDK static lib architecture to validate (default: auto=host arch).",
    )
    args = parser.parse_args()

    default_sdk_root = _repo_root() / "third_party" / "unitree_sdk2"
    sdk_root = Path(args.sdk_root).expanduser().resolve() if args.sdk_root else default_sdk_root

    print("=== Unitree SDK2 Setup Check ===")
    _print_expected_layout(sdk_root)

    ok, messages = validate_sdk_root(sdk_root, target_arch=str(args.target_arch))
    for line in messages:
        print(line)

    print("")
    print("Next steps:")
    print(f"  1) Place SDK source in: {sdk_root} (current default)")
    print("  2) Optional prune: python3 unitree_go2_realobs/scripts/real/prune_sdk2_for_go2.py --apply")
    print("  3) Build bridge: cmake -S unitree_go2_realobs/scripts/real/sdk2_bridge -B /tmp/go2_udp_bridge_build && cmake --build /tmp/go2_udp_bridge_build -j")
    print("  4) Run go2_udp_bridge + run_governor_live_template.py (see scripts/real/README.md).")

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
