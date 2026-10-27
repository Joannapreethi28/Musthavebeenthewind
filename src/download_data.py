"""Download only the agreed NOAA/NASA products; accept each file after validation."""

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from data_validation import MAX_DATA_BYTES, require, validate_goes, validate_omni

PROJECT = Path(__file__).resolve().parents[1]
GOES_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/"
OMNI_BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
HOSTS = {"www.ncei.noaa.gov", "spdf.gsfc.nasa.gov"}
GOES_NAME = re.compile(r"g15_epead_e13ew_5m_\d{8}_\d{8}\.nc")
LISTING_LIMIT = 2 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_url(url: str, expected_host: str | None = None) -> str:
    """Keep both the initial request and every redirect on an official HTTPS host."""
    parsed = urlsplit(url)
    require(parsed.scheme == "https" and parsed.hostname in HOSTS, f"Unapproved source URL: {url}")
    require(parsed.username is None and parsed.password is None and parsed.port in (None, 443), "Unexpected URL credentials or port")
    require(not parsed.query and not parsed.fragment, "Unexpected URL query or fragment")
    require(expected_host is None or parsed.hostname == expected_host, "Redirect or link changed source host")
    return url


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        check_url(newurl, urlsplit(request.full_url).hostname)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class ListingLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def expected_name(year: int, month: int) -> str:
    return f"g15_epead_e13ew_5m_{year}{month:02d}01_{year}{month:02d}{calendar.monthrange(year, month)[1]:02d}.nc"


def select_month_url(html: str, listing_url: str, year: int, month: int) -> str:
    parser = ListingLinks()
    parser.feed(html)
    matches = set()
    for href in parser.links:
        url = urljoin(listing_url, href)
        if GOES_NAME.fullmatch(urlsplit(url).path.rsplit("/", 1)[-1]):
            matches.add(check_url(url, "www.ncei.noaa.gov"))
    require(len(matches) == 1, f"Expected one monthly e13ew 5m file; found {len(matches)}")
    url = matches.pop()
    require(urlsplit(url).path.rsplit("/", 1)[-1] == expected_name(year, month), "Listing contains the wrong month/date range")
    require(url.rsplit("/", 1)[0] + "/" == listing_url, "Data link leaves its monthly directory")
    return url


def fetch(url: str, opener, limit: int, destination: Path | None = None) -> tuple[bytes, str]:
    """Bound response size and retries; keep TLS verification enabled."""
    check_url(url)
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "GEO-electron-flux-course-project/1.0", "Accept-Encoding": "identity"})
            with opener.open(request, timeout=30) as response:
                final_url = check_url(response.geturl(), urlsplit(url).hostname)
                require(response.status == 200, f"Unexpected HTTP status {response.status}")
                require(response.headers.get("Content-Encoding", "identity").lower() == "identity", "Unexpected compressed response")
                length = response.headers.get("Content-Length")
                if length is not None:
                    require(0 < int(length) <= limit, "Response declares an empty or oversized body")
                # At most 8 MiB per data file; a bounded buffer keeps partial content out of raw/.
                content = response.read(limit + 1)
                require(0 < len(content) <= limit, "Response is empty or exceeds size limit")
                require(length is None or len(content) == int(length), "Truncated response body")
                if destination is not None:
                    destination.write_bytes(content)
                return content, final_url
        except (URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {url}: {error}") from error
            print(f"Retry {attempt + 1}/2: {url}", flush=True)
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable retry state")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def accept_file(path: Path, url: str, validator, opener) -> dict:
    """Validate a temporary file before making it visible as accepted raw data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        _, final_url = fetch(url, opener, MAX_DATA_BYTES, temporary)
        validation = validator(temporary)
        digest = sha256(temporary)
        # An orphan from an interrupted manifest write is only adopted if its bytes match.
        require(not path.exists() or sha256(path) == digest, f"Existing file differs from source; preserved {path}")
        entry = {"source_url": url, "final_url": final_url, "retrieved_at_utc": utc_now(),
                 "size_bytes": temporary.stat().st_size, "sha256": digest, "validation": validation}
        temporary.replace(path)
        return entry
    finally:
        temporary.unlink(missing_ok=True)


def run(verify_only: bool = False, project: Path = PROJECT) -> dict:
    """Resume safely; a complete cached run performs no network requests."""
    project = project.resolve()
    raw = project / "data" / "raw"
    require(raw.resolve().is_relative_to(project), "Raw data directory resolves outside project")
    manifest_path = project / "results" / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("schema_version") == 1, "Unsupported data manifest schema")
    else:
        manifest = {"schema_version": 1, "created_at_utc": utc_now(), "files": {}}
    manifest["status"] = "incomplete"
    manifest.pop("last_successful_run", None)
    files = manifest["files"]
    jobs = [("goes15", year, month) for year in range(2012, 2018) for month in range(1, 13)]
    jobs += [("omni2", year, None) for year in range(2012, 2018)]
    expected_paths = {
        f"data/raw/{kind}/{expected_name(year, month) if month else f'omni2_{year}.dat'}"
        for kind, year, month in jobs
    }
    require(set(files).issubset(expected_paths), "Manifest contains files outside the agreed 78-file dataset")
    opener = build_opener(OfficialRedirects())
    downloaded = skipped = 0
    save_manifest(manifest_path, manifest)
    def process(job):
        kind, year, month = job
        opener = build_opener(OfficialRedirects())
        name = expected_name(year, month) if month else f"omni2_{year}.dat"
        path = raw / kind / name
        require(path.resolve().is_relative_to(raw.resolve()), "Data path resolves outside raw directory")
        key = path.relative_to(project).as_posix()
        validator = (lambda p: validate_goes(p, year, month)) if month else (lambda p: validate_omni(p, year))
        entry = files.get(key)
        if path.exists() and entry:
            require(path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"],
                    f"Integrity mismatch; preserved {key}. Investigate before replacing it.")
            check_url(entry["source_url"], "www.ncei.noaa.gov" if month else "spdf.gsfc.nasa.gov")
            check_url(entry["final_url"], "www.ncei.noaa.gov" if month else "spdf.gsfc.nasa.gov")
            entry["validation"] = validator(path)
            action = "VERIFIED"
        else:
            require(not verify_only, f"Missing file or provenance: {key}; run download mode first")
            if month:
                listing = f"{GOES_BASE}{year}/{month:02d}/goes15/netcdf/"
                html, final_listing = fetch(listing, opener, LISTING_LIMIT)
                url = select_month_url(html.decode("utf-8"), final_listing, year, month)
            else:
                url = f"{OMNI_BASE}{name}"
            entry = accept_file(path, url, validator, opener)
            action = "DOWNLOADED + VERIFIED"
        entry["last_verified_at_utc"] = utc_now()
        return key, entry, action, name
    # Four bounded workers reduce archive latency; only this thread writes provenance.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for number, (key, entry, action, name) in enumerate(executor.map(process, jobs), 1):
            files[key] = entry
            downloaded += int(action == 'DOWNLOADED + VERIFIED')
            skipped += int(action == 'VERIFIED')
            save_manifest(manifest_path, manifest)
            print(f"[{number:02d}/78] {action}: {name}", flush=True)
    require(set(files) == expected_paths, "Missing expected source files")
    actual_paths = {p.relative_to(project).as_posix() for p in raw.rglob("*") if p.is_file()}
    require(actual_paths == expected_paths, "Unexpected extra/partial files in data/raw; inspect before proceeding")
    summary = {"goes_files": 72, "omni_files": 6,
               "goes_rows": sum(v["validation"]["rows"] for k, v in files.items() if "/goes15/" in k),
               "omni_rows": sum(v["validation"]["rows"] for k, v in files.items() if "/omni2/" in k),
               "total_bytes": sum(v["size_bytes"] for v in files.values()),
               "downloaded_this_run": downloaded, "verified_existing_this_run": skipped,
               "completed_at_utc": utc_now(), "mode": "verify-only" if verify_only else "download"}
    summary['goes_absent_grid_records'] = sum(v['validation']['absent_grid_records'] for k, v in files.items() if '/goes15/' in k)
    require(summary["goes_rows"] + summary['goes_absent_grid_records'] == 631296 and summary["omni_rows"] == 52608, "Unexpected total time coverage")
    manifest.update(status="complete", last_successful_run=summary)
    save_manifest(manifest_path, manifest)
    print("PASS: 72 GOES + 6 OMNI files; 2012-2017 coverage audited, absent GOES records counted; SHA-256 recorded.", flush=True)
    print(f"Downloaded: {downloaded}; verified existing: {skipped}; bytes: {summary['total_bytes']:,}.", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check all files and saved hashes without accessing the network")
    arguments = parser.parse_args()
    try:
        run(arguments.verify_only)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"STOP: {error}\nNo preprocessing or training was started.\n")
