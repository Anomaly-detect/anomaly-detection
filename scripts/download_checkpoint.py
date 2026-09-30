#!/usr/bin/env python3
"""
Checkpoint Downloader for Video Anomaly Detection
-------------------------------------------------
Downloads pretrained model weights (e.g., wider_resnet38.pth) from MinIO / S3 storage.
Supports:
  - Automatic MinIO Console browser URL conversion to direct S3 download URL
  - MD5 integrity check (skips download if valid file already exists)
  - Streaming download with real-time progress bar (percentage, MB, speed, ETA)
  - Atomic writing via temporary file to prevent file corruption
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

DEFAULT_CHECKPOINT_URL = "https://minio1.webtui.vn:9000/browser/bucket-vmt/checkpoint%2Fwider_resnet38.pth"
DEFAULT_MD5 = "64326bc674b23d7015924a135eca3ae9"
DEFAULT_FILENAME = "wider_resnet38.pth"


def normalize_download_url(url: str) -> str:
    """
    Normalizes a MinIO URL.
    Converts MinIO browser UI URLs:
      https://host:port/browser/bucket/path%2Ffile.ext
    Into direct S3 object download URLs:
      https://host:port/bucket/path/file.ext
    """
    parsed = urllib.parse.urlsplit(url.strip())
    path = parsed.path
    if "/browser/" in path:
        path = path.replace("/browser/", "/", 1)
        path = urllib.parse.unquote(path)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def compute_file_md5(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Computes the MD5 hex digest of a local file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def format_size(bytes_num: float) -> str:
    """Formats bytes into human readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_num) < 1024.0:
            return f"{bytes_num:3.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"


def format_time(seconds: float) -> str:
    """Formats seconds into readable mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 0:
        return "--:--"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def download_checkpoint(
    url: str = DEFAULT_CHECKPOINT_URL,
    dest: str = "checkpoints",
    expected_md5: str = DEFAULT_MD5,
    force: bool = False,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Downloads a checkpoint file with verification and progress reporting.

    Args:
        url: Source download URL (supports MinIO browser UI or direct URLs).
        dest: Destination path (directory or target file path).
        expected_md5: Expected MD5 hash for verification.
        force: If True, re-downloads even if the file exists and is valid.
        timeout: Network timeout in seconds.
        chunk_size: Chunk size in bytes for streaming download.

    Returns:
        The absolute path to the downloaded file.
    """
    download_url = normalize_download_url(url)
    
    # Determine target filename and directory
    url_filename = os.path.basename(urllib.parse.urlsplit(download_url).path)
    if not url_filename:
        url_filename = DEFAULT_FILENAME

    dest = os.path.abspath(dest)
    if os.path.isdir(dest) or not os.path.splitext(dest)[1]:
        dest_dir = dest
        target_file = os.path.join(dest_dir, url_filename)
    else:
        dest_dir = os.path.dirname(dest)
        target_file = dest

    os.makedirs(dest_dir, exist_ok=True)

    print("==================================================")
    print(" Video Anomaly Detection - Checkpoint Downloader")
    print("==================================================")
    print(f" Source URL   : {download_url}")
    print(f" Destination  : {target_file}")
    if expected_md5:
        print(f" Expected MD5 : {expected_md5}")
    print("--------------------------------------------------")

    # Step 1: Check if file already exists and is valid
    if os.path.isfile(target_file) and not force:
        local_size = os.path.getsize(target_file)
        print(f"[*] Found existing file ({format_size(local_size)}). Verifying checksum...")
        local_md5 = compute_file_md5(target_file)
        if expected_md5 and local_md5.lower() == expected_md5.lower():
            print(f"[✓] Checkpoint already exists and matches expected MD5: {local_md5}")
            print(f"[✓] Path: {target_file}")
            print("==================================================")
            return target_file
        elif expected_md5:
            print(f"[!] Checksum mismatch for existing file:")
            print(f"    Existing: {local_md5}")
            print(f"    Expected: {expected_md5}")
            print("[*] Re-downloading checkpoint...")
        else:
            print(f"[✓] Existing file verified (MD5: {local_md5})")
            return target_file

    # Step 2: Prepare request and connect
    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "AnomalyDetection-CheckpointDownloader/1.0"},
    )

    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed to connect to {download_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Check total size and server ETag if available
    content_length = response.headers.get("Content-Length")
    total_size = int(content_length) if content_length and content_length.isdigit() else 0
    server_etag = response.headers.get("ETag", "").strip('"')
    target_md5 = expected_md5 or server_etag or None

    temp_file = target_file + ".tmp"
    hasher = hashlib.md5()
    downloaded_bytes = 0
    start_time = time.time()
    last_print_time = 0.0

    print(f"[*] Downloading: {os.path.basename(target_file)} ({format_size(total_size) if total_size else 'Unknown size'})")

    try:
        with open(temp_file, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                hasher.update(chunk)
                downloaded_bytes += len(chunk)

                current_time = time.time()
                # Update progress bar at most 10 times a second
                if current_time - last_print_time > 0.1 or downloaded_bytes == total_size:
                    elapsed = current_time - start_time
                    speed = downloaded_bytes / elapsed if elapsed > 0 else 0
                    if total_size > 0:
                        pct = (downloaded_bytes / total_size) * 100
                        remaining_bytes = total_size - downloaded_bytes
                        eta = remaining_bytes / speed if speed > 0 else 0
                        bar_len = 30
                        filled = int(bar_len * downloaded_bytes // total_size)
                        bar = "=" * filled + (">" if filled < bar_len else "") + "." * (bar_len - filled - 1)
                        progress_msg = (
                            f"\r[{bar[:bar_len]}] {pct:5.1f}% | "
                            f"{format_size(downloaded_bytes)}/{format_size(total_size)} | "
                            f"{format_size(speed)}/s | ETA: {format_time(eta)}"
                        )
                    else:
                        progress_msg = (
                            f"\rDownloaded: {format_size(downloaded_bytes)} | "
                            f"{format_size(speed)}/s"
                        )
                    sys.stdout.write(progress_msg)
                    sys.stdout.flush()
                    last_print_time = current_time

        sys.stdout.write("\n")
        sys.stdout.flush()

    except (KeyboardInterrupt, Exception) as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if isinstance(e, KeyboardInterrupt):
            print("\n[!] Download cancelled by user.", file=sys.stderr)
        else:
            print(f"\n[ERROR] Download interrupted: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Checksum verification
    downloaded_md5 = hasher.hexdigest()
    if target_md5 and downloaded_md5.lower() != target_md5.lower():
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"[ERROR] MD5 checksum verification failed!", file=sys.stderr)
        print(f"        Downloaded: {downloaded_md5}", file=sys.stderr)
        print(f"        Expected  : {target_md5}", file=sys.stderr)
        sys.exit(1)

    # Step 4: Atomic file move
    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(temp_file, target_file)

    total_time = time.time() - start_time
    avg_speed = downloaded_bytes / total_time if total_time > 0 else 0
    print(f"[✓] Download completed successfully in {format_time(total_time)} ({format_size(avg_speed)}/s)")
    print(f"[✓] MD5: {downloaded_md5}")
    print(f"[✓] Saved to: {target_file}")
    print("==================================================")
    return target_file


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights from MinIO storage for Video Anomaly Detection."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_CHECKPOINT_URL,
        help="Download URL (default: %(default)s)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="checkpoints",
        help="Destination directory or file path (default: %(default)s)",
    )
    parser.add_argument(
        "--md5",
        type=str,
        default=DEFAULT_MD5,
        help="Expected MD5 checksum (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists and passes MD5 check",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()
    download_checkpoint(
        url=args.url,
        dest=args.dest,
        expected_md5=args.md5,
        force=args.force,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
