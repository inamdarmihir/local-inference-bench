"""
searchai_client.py

Install/run/probe helpers for SearchBlox's SearchAI Index Server, written
as functions (not just notebook cells) so the sequence is reviewable and
reusable outside Colab. Every probe returns the raw HTTP status/body
alongside its verdict -- nothing here infers or assumes a result the
server didn't actually return.

This module shares no code with corpus.py/tokens.py/pricing.py/
run_local_benchmark.py/push_to_qdrant.py: it talks to a separate external
process (the SearchAI Index Server binary) over HTTP, not to fastembed or
an in-process qdrant-client.

Verified facts this module builds on (confirmed by reading the actual
install script and SearchBlox's own documentation, not assumed):
  - The install is Linux-only; default port 9200, default install prefix
    /opt/searchai-index.
  - Two install paths exist: the one-line installer (curl | sudo bash),
    which sets up a systemd service, and a documented manual fallback --
    extract the release tarball and run ./run.sh in the foreground.
    Environments without a full init system (containers, Colab) should
    prefer the foreground path; this module implements that path.
  - The Qdrant dialect listener is off by default. Enabling it means
    appending dialects.qdrant-port=<port> to
    <prefix>/conf/server.properties and restarting.
  - No published documentation shows a plain-text (vector-less) upsert
    body for the Qdrant dialect. Whether the dialect embeds text
    server-side at ingest, versus requiring a client-supplied vector like
    real Qdrant, is the single open question this module exists to answer
    empirically -- see probe_plaintext_upsert().
"""

import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VERSION = "1.3.0"
DEFAULT_PREFIX = Path("/opt/searchai-index")
DEFAULT_PORT = 9200
DEFAULT_QDRANT_PORT = 6333
RELEASE_BASE_URL = "https://index-server.searchblox.com"


@dataclass
class ProbeResult:
    name: str
    ok: bool
    status_code: int | None
    request: dict | None
    response_body: str


def download_and_unpack(
    version: str = DEFAULT_VERSION,
    arch: str = "amd64",
    work_dir: Path = Path("/content"),
) -> Path:
    """Downloads and unpacks the same release tarball the official
    one-line installer fetches, without running its systemd install
    step. Returns the unpacked bundle directory."""
    tarball = f"searchai-index-server-{version}-linux-{arch}.tar.gz"
    url = f"{RELEASE_BASE_URL}/releases/{version}/{tarball}"
    dest = work_dir / tarball
    urllib.request.urlretrieve(url, dest)
    subprocess.run(["tar", "-xzf", str(dest), "-C", str(work_dir)], check=True)
    unpacked_name = (
        subprocess.run(
            ["tar", "-tzf", str(dest)], capture_output=True, text=True, check=True
        )
        .stdout.splitlines()[0]
        .split("/")[0]
    )
    return work_dir / unpacked_name


def start_foreground(bundle_dir: Path, log_path: Path) -> subprocess.Popen:
    """Launches <bundle_dir>/run.sh in the background (not systemd),
    logging stdout/stderr to log_path. Colab (and most containers) have
    no full init system to reliably supervise a systemd service, so this
    is the primary launch path; the full installer is only a documented
    fallback attempt."""
    log_file = open(log_path, "w")
    return subprocess.Popen(
        ["./run.sh"], cwd=str(bundle_dir), stdout=log_file, stderr=subprocess.STDOUT
    )


def wait_for_health(
    port: int = DEFAULT_PORT,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
    timeout_s: int = 150,
    interval_s: int = 5,
) -> bool:
    """Polls http://localhost:<port>/ until it responds (status < 500) or
    timeout_s elapses. If proc exits early, or the timeout is hit, prints
    the process's own log tail and returns False -- never hangs silently
    or retries past the stated budget."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            print(f"run.sh exited early with code {proc.returncode}")
            if log_path and log_path.exists():
                print(log_path.read_text()[-4000:])
            return False
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=3) as resp:
                if resp.status < 500:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval_s)
    print(f"Timed out after {timeout_s}s waiting for http://localhost:{port}/")
    if log_path and log_path.exists():
        print(log_path.read_text()[-4000:])
    return False


def find_admin_api_key(log_path: Path) -> str | None:
    """A fresh install auto-generates and prints an admin API key (per
    SearchBlox's own published example). Looks for it in the launch log.
    Returns None if not found -- callers should then try unauthenticated
    requests, since auth is documented as off by default until
    server.api-key is set."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    match = re.search(
        r"(?:api[-_ ]?key)\s*[:=]\s*([A-Za-z0-9_\-]{16,})", text, re.IGNORECASE
    )
    return match.group(1) if match else None


def enable_qdrant_dialect(
    prefix: Path = DEFAULT_PREFIX, qdrant_port: int = DEFAULT_QDRANT_PORT
) -> None:
    """Appends dialects.qdrant-port=<qdrant_port> to server.properties if
    not already present. Caller is responsible for restarting the server
    afterwards -- this alone has no effect on an already-running process."""
    props_path = prefix / "conf" / "server.properties"
    text = props_path.read_text() if props_path.exists() else ""
    if "dialects.qdrant-port" not in text:
        with open(props_path, "a") as f:
            f.write(f"\ndialects.qdrant-port={qdrant_port}\n")


def _request(
    method: str, url: str, body: dict | None = None, api_key: str | None = None
) -> ProbeResult:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return ProbeResult(
                name=url,
                ok=resp.status < 400,
                status_code=resp.status,
                request=body,
                response_body=resp.read().decode(errors="replace"),
            )
    except urllib.error.HTTPError as e:
        return ProbeResult(
            name=url,
            ok=False,
            status_code=e.code,
            request=body,
            response_body=e.read().decode(errors="replace"),
        )
    except urllib.error.URLError as e:
        return ProbeResult(
            name=url, ok=False, status_code=None, request=body, response_body=str(e)
        )


def probe_version(port: int = DEFAULT_PORT, api_key: str | None = None) -> ProbeResult:
    return _request("GET", f"http://localhost:{port}/", api_key=api_key)


def probe_create_qdrant_collection(
    collection: str, dim: int, qdrant_port: int = DEFAULT_QDRANT_PORT
) -> ProbeResult:
    """Known-good per SearchBlox's own docs (a client-supplied-vector
    collection create): establishes the Qdrant dialect responds at all,
    before testing the riskier vector-less-upsert case below."""
    body = {"vectors": {"size": dim, "distance": "Cosine"}}
    return _request(
        "PUT", f"http://localhost:{qdrant_port}/collections/{collection}", body=body
    )


def probe_plaintext_upsert(
    collection: str, text: str, qdrant_port: int = DEFAULT_QDRANT_PORT
) -> ProbeResult:
    """THE critical probe. Sends a point with no "vector" key at all --
    only a payload -- to test whether the Qdrant dialect embeds text
    server-side at ingest (the premise the whole SearchAI leg depends on)
    or requires a client-supplied vector like real Qdrant. No published
    documentation shows this working either way; a rejection here is a
    valid, expected, and reportable result -- not a bug to route around."""
    body = {"points": [{"id": 1, "payload": {"body": text}}]}
    return _request(
        "PUT",
        f"http://localhost:{qdrant_port}/collections/{collection}/points",
        body=body,
    )


def probe_opensearch_autoembed(
    index: str,
    text: str,
    model: str = "bge-small-en-v1.5",
    dim: int = 384,
    port: int = DEFAULT_PORT,
) -> tuple[ProbeResult, ProbeResult]:
    """The alternate, independently-documented path: an OpenSearch index
    with a knn_vector field carrying an "embed" block naming the source
    field and model (per SearchBlox's own published mapping example).
    Creates that mapping, then indexes one plain-text document.

    A pass here does NOT validate the Qdrant-dialect premise tested by
    probe_plaintext_upsert() -- it is a different API surface entirely and
    must be reported as its own, separately-labeled result, never
    substituted for the Qdrant-dialect probe."""
    mapping = {
        "mappings": {
            "properties": {
                "body": {"type": "text"},
                "vec": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "space_type": "cosinesimil",
                    "embed": {"source_field": "body", "model": model},
                },
            }
        }
    }
    create = _request("PUT", f"http://localhost:{port}/{index}", body=mapping)
    index_doc = _request(
        "POST", f"http://localhost:{port}/{index}/_doc/1", body={"body": text}
    )
    return create, index_doc


def sample_gpu_utilization(
    stop_event: threading.Event, interval_s: float = 1.0
) -> list[int]:
    """Samples `nvidia-smi --query-gpu=utilization.gpu` once per
    interval_s until stop_event is set. A coarse proxy -- 1Hz sampling can
    miss short bursts -- so pair with process_uses_cuda() below for a
    second, independent signal rather than trusting this alone."""
    samples: list[int] = []
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            if out.isdigit():
                samples.append(int(out))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(interval_s)
    return samples


def process_uses_cuda(pid: int) -> bool:
    """Second, independent signal for GPU use: checks whether the process
    has any CUDA/NVIDIA library mapped into its address space. Disagreement
    with the utilization sampler should be reported as inconclusive, not
    resolved by silently picking one signal over the other."""
    try:
        maps = Path(f"/proc/{pid}/maps").read_text()
        return "cuda" in maps.lower() or "nvidia" in maps.lower()
    except (FileNotFoundError, PermissionError):
        return False


def timed_ingest(upsert_fn, documents: list[str], server_pid: int | None = None) -> dict:
    """Times upsert_fn(documents) -- a caller-supplied closure over
    whichever endpoint validated (Qdrant-dialect or OpenSearch-autoembed)
    -- while concurrently sampling GPU utilization, so "did this use the
    GPU" is an empirical result rather than an assumption from the runtime
    type alone."""
    stop_event = threading.Event()
    samples: list[int] = []

    def _collect():
        samples.extend(sample_gpu_utilization(stop_event))

    sampler = threading.Thread(target=_collect)
    sampler.start()

    start = time.perf_counter()
    upsert_fn(documents)
    elapsed = time.perf_counter() - start

    stop_event.set()
    sampler.join()

    max_util = max(samples, default=0)
    mean_util = sum(samples) / len(samples) if samples else 0.0
    cuda_loaded = process_uses_cuda(server_pid) if server_pid else None

    return {
        "documents_ingested": len(documents),
        "wall_clock_seconds": elapsed,
        "documents_per_second": len(documents) / elapsed if elapsed > 0 else 0.0,
        "gpu_utilization_max_pct": max_util,
        "gpu_utilization_mean_pct": mean_util,
        "gpu_utilization_samples": len(samples),
        "process_maps_show_cuda": cuda_loaded,
        "gpu_acceleration_observed": bool(max_util > 5 or cuda_loaded),
    }
