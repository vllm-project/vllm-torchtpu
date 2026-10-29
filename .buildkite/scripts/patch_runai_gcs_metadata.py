# SPDX-License-Identifier: Apache-2.0
"""Temporary Decode CI mitigation for RunAI GCS metadata read/write races.

RunAI 0.17.0's runai_model_streamer_gcs/files/files.py downloads metadata
directly into shared cache files. Publish each completed download atomically
before starting API servers. Remove this patch when the dependency fixes it.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

_SOURCE_SHA256 = "1724ba702bd69b023ac5a2a4ec9914af238648c3adb017eb9ebc1f1b22b8370d"


def patch_source(source: str) -> str:
    """Reject unreviewed dependency changes rather than silently mispatching."""
    if hashlib.sha256(source.encode()).hexdigest() != _SOURCE_SHA256:
        raise RuntimeError("RunAI GCS source changed; review or remove the CI patch")
    source = source.replace("import posixpath\n", "import posixpath\nimport tempfile\n")
    return source.replace(
        "        blob.download_to_filename(destination_file)\n",
        """        # Readers must not see another process's incomplete download.
        fd, temporary_file = tempfile.mkstemp(dir=local_dir, prefix=".runai-")
        os.close(fd)
        try:
            blob.download_to_filename(temporary_file)
            os.replace(temporary_file, destination_file)
        finally:
            Path(temporary_file).unlink(missing_ok=True)
""",
    )


def main() -> None:
    dist = importlib.metadata.distribution("runai-model-streamer-gcs")
    if dist.version != "0.17.0":
        raise RuntimeError(f"Review the RunAI GCS CI patch for version {dist.version}")
    path = Path(dist.locate_file("runai_model_streamer_gcs/files/files.py"))
    patched = patch_source(path.read_text())
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(
        json.dumps(
            {
                "package": "runai-model-streamer-gcs",
                "version": dist.version,
                "source_sha256": _SOURCE_SHA256,
                "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
