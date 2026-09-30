"""
Checkpoint downloader utility alias.
Forwards to scripts/download_checkpoint.py.
"""

import os
import sys

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.download_checkpoint import (
    DEFAULT_CHECKPOINT_URL,
    DEFAULT_MD5,
    DEFAULT_FILENAME,
    normalize_download_url,
    compute_file_md5,
    download_checkpoint,
    main,
)

__all__ = [
    "DEFAULT_CHECKPOINT_URL",
    "DEFAULT_MD5",
    "DEFAULT_FILENAME",
    "normalize_download_url",
    "compute_file_md5",
    "download_checkpoint",
    "main",
]

if __name__ == "__main__":
    main()
