"""Print the test files one shard of a recording step runs, one per line.

PROTOTYPE. Every file CI's test-file pattern matches
(<tests dir>/**/{test_*,*_test}.py), minus the excluded ones, dealt
round-robin.

    python3 shard_files.py [--tests-dir DIR] SHARD SHARDS [EXCLUDED_FILE...]
"""

import pathlib
import sys


def shard_files(root: pathlib.Path, shard: int, shards: int, excluded: set[str],
                tests_dir: str = "tests") -> list[str]:
    files = sorted(
        {
            p.relative_to(root).as_posix()
            for pattern in ("test_*.py", "*_test.py")
            for p in (root / tests_dir).rglob(pattern)
        }
        - excluded
    )
    return files[shard::shards]


if __name__ == "__main__":
    argv = sys.argv[1:]
    tests_dir = "tests"
    if argv[:1] == ["--tests-dir"]:
        tests_dir, argv = argv[1], argv[2:]
    shard, shards, *excluded = argv
    for f in shard_files(pathlib.Path.cwd(), int(shard), int(shards), set(excluded),
                         tests_dir):
        print(f)
