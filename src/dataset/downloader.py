from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from src.utils.config import project_path
from src.utils.files import safe_relative_path, sha256_file, slugify


LOGGER = logging.getLogger(__name__)
API_ROOT = "https://api.data.mendeley.com"
USER_AGENT = "MangoResearch-Phase0/1.0"
PUBLIC_ZIP_ROOT = "https://data.mendeley.com/public-api/zip"


class _StripCrossOriginAuthorization(urllib.request.HTTPRedirectHandler):
    """Do not forward an OAuth bearer token to a signed storage host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            redirected.remove_header("Authorization")
            redirected.remove_header("authorization")
        return redirected


class DownloadError(RuntimeError):
    """Raised when the official source cannot be accessed or downloaded."""


class AuthenticationRequired(DownloadError):
    """Raised when Mendeley refuses an API request without valid OAuth2."""


class MendeleyDownloader:
    def __init__(self, config: dict[str, Any], retries: int = 3, timeout: int = 60):
        self.config = config
        self.retries = max(1, retries)
        self.timeout = timeout
        self.token = os.environ.get("MENDELEY_ACCESS_TOKEN", "").strip()
        self.opener = urllib.request.build_opener(_StripCrossOriginAuthorization())

    def _open(self, url: str):
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise AuthenticationRequired(
                        f"Mendeley API từ chối truy cập (HTTP {exc.code}). "
                        "API yêu cầu OAuth2; đặt MENDELEY_ACCESS_TOKEN hợp lệ hoặc "
                        "tải thủ công từ trang dataset chính thức."
                    ) from exc
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise DownloadError(f"Mendeley API trả HTTP {exc.code}: {url}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                delay = min(2 ** attempt, 8)
                LOGGER.warning("Request lỗi, thử lại sau %ss (%s)", delay, url)
                time.sleep(delay)
        raise DownloadError(f"Không truy cập được Mendeley sau {self.retries} lần: {last_error}")

    def _json(self, url: str) -> Any:
        try:
            with self._open(url) as response:
                return json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DownloadError(f"Phản hồi API không phải JSON: {url}") from exc

    @staticmethod
    def _items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("files", "folders", "results", "items", "data"):
                if isinstance(payload.get(key), list):
                    return [item for item in payload[key] if isinstance(item, dict)]
        return []

    def _url(self, dataset_id: str, endpoint: str, params: dict[str, str] | None = None) -> str:
        url = f"{API_ROOT}/datasets/{dataset_id}/{endpoint.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def _list_files(self, dataset_id: str, version: int) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        folders: list[tuple[str | None, str]] = [(None, "")]
        seen_folders: set[str] = set()
        while folders:
            parent_id, parent_path = folders.pop()
            folder_params = {"version": str(version)}
            if parent_id:
                folder_params["parent_id"] = parent_id
            folder_payload = self._json(self._url(dataset_id, "folders", folder_params))
            for folder in self._items(folder_payload):
                folder_id = str(folder.get("id") or "")
                folder_name = str(folder.get("name") or "")
                if not folder_id or folder_id in seen_folders or not folder_name:
                    continue
                seen_folders.add(folder_id)
                relative = (Path(parent_path) / folder_name).as_posix().strip(".")
                folders.append((folder_id, relative))

            start = 0
            while True:
                params = {"version": str(version), "$start": str(start), "$limit": "100"}
                if parent_id:
                    params["folder_id"] = parent_id
                payload = self._json(self._url(dataset_id, "files", params))
                batch = self._items(payload)
                for item in batch:
                    item = dict(item)
                    item["_relative_folder"] = parent_path
                    files.append(item)
                if len(batch) < 100:
                    break
                start += len(batch)
        # Some API deployments return all files from the root listing and do
        # not populate folder listing. Keep that response without duplicates.
        unique: dict[str, dict[str, Any]] = {}
        for item in files:
            if item.get("id"):
                unique[str(item["id"])] = item
        return list(unique.values())

    @staticmethod
    def _matches(item: dict[str, Any], include: list[str], exclude: list[str]) -> bool:
        name = str(item.get("_relative_folder", "")) + "/" + str(item.get("filename", ""))
        normalized = name.casefold()
        included = not include or any(fnmatch.fnmatch(normalized, p.casefold()) for p in include)
        excluded = any(fnmatch.fnmatch(normalized, p.casefold()) for p in exclude)
        return included and not excluded

    def list_remote(self, dataset_key: str) -> list[dict[str, Any]]:
        definition = self.config["datasets"][dataset_key]
        files = self._list_files(str(definition["dataset_id"]), int(definition.get("version", 1)))
        return files

    def download_public_zips(self, dataset_keys: list[str] | None = None) -> dict[str, Path]:
        """Download the same public ZIP exposed by each dataset page's Download All button."""
        keys = dataset_keys or list(self.config["datasets"])
        result: dict[str, Path] = {}
        for key in keys:
            if key not in self.config["datasets"]:
                raise ValueError(f"Dataset không có trong config: {key}")
            definition = self.config["datasets"][key]
            dataset_id = urllib.parse.quote(str(definition["dataset_id"]), safe="")
            version = int(definition.get("version", 1))
            root = project_path(self.config, f"data/raw/{key}/downloads")
            root.mkdir(parents=True, exist_ok=True)
            target = root / str(definition.get("archive_name") or f"{key}_v{version}.zip")
            partial = target.with_name(target.name + ".part")
            if target.exists():
                if target.stat().st_size > 0 and zipfile.is_zipfile(target):
                    LOGGER.info("Đã có public ZIP hợp lệ, bỏ qua: %s", target)
                    result[key] = target
                    continue
                raise DownloadError(f"Đích đã tồn tại nhưng không phải ZIP hợp lệ; giữ nguyên để tránh ghi đè: {target}")

            if partial.exists() and partial.stat().st_size:
                with partial.open("rb") as stream:
                    if stream.read(4)[:2] != b"PK":
                        raise DownloadError(f"File .part không có chữ ký ZIP; giữ nguyên để kiểm tra thủ công: {partial}")

            url = f"{PUBLIC_ZIP_ROOT}/{dataset_id}/download/{version}"
            last_error: Exception | None = None
            for attempt in range(self.retries):
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {
                    "User-Agent": USER_AGENT,
                    "Accept": "application/zip, application/octet-stream",
                }
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                try:
                    response = self.opener.open(request, timeout=self.timeout)
                    with response:
                        status = response.getcode()
                        final_scheme = urllib.parse.urlsplit(response.geturl()).scheme.casefold()
                        if final_scheme != "https":
                            raise DownloadError("Download All bị chuyển hướng khỏi HTTPS; đã dừng.")
                        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
                        if content_type not in {"application/zip", "application/octet-stream", "application/x-zip-compressed", "binary/octet-stream"}:
                            raise DownloadError(
                                f"Download All không trả về ZIP (HTTP {status}, Content-Type={content_type or 'missing'}); "
                                "không lưu phản hồi lỗi thành archive."
                            )

                        content_range = response.headers.get("Content-Range", "")
                        if status == 206:
                            match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip(), flags=re.IGNORECASE)
                            if not match or int(match.group(1)) != offset:
                                raise DownloadError("Server trả Content-Range không khớp offset; không nối vào file .part.")
                            mode = "ab" if offset else "wb"
                        elif status == 200:
                            # Server ignored Range; safely restart the partial file from byte zero.
                            offset = 0
                            mode = "wb"
                        else:
                            raise DownloadError(f"Download All trả HTTP {status}; không lưu phản hồi thành ZIP.")

                        expected_length = int(response.headers["Content-Length"]) if response.headers.get("Content-Length", "").isdigit() else 0
                        received = 0
                        with partial.open(mode) as output:
                            while chunk := response.read(1024 * 1024):
                                if offset == 0 and received == 0 and not chunk.startswith(b"PK"):
                                    raise DownloadError("Phản hồi không có chữ ký ZIP; không lưu nội dung.")
                                output.write(chunk)
                                received += len(chunk)
                        if expected_length and received != expected_length:
                            raise OSError(f"Kết nối bị ngắt trước Content-Length ({received}/{expected_length})")

                    if not zipfile.is_zipfile(partial):
                        raise DownloadError(f"Tải xong nhưng file không phải ZIP hợp lệ; giữ .part để kiểm tra: {partial}")
                    try:
                        with zipfile.ZipFile(partial) as bundle:
                            bundle.infolist()
                    except zipfile.BadZipFile as exc:
                        raise DownloadError(f"Central directory của ZIP không hợp lệ: {partial}") from exc
                    if target.exists():
                        raise DownloadError(f"Đích xuất hiện trong khi tải; không ghi đè: {target}")
                    partial.replace(target)
                    LOGGER.info("Đã tải public Download All ZIP: %s (%s byte)", target, target.stat().st_size)
                    result[key] = target
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code in {401, 403}:
                        raise AuthenticationRequired(
                            f"Public Download All trả HTTP {exc.code}; đã dừng, không thử vượt xác thực/CAPTCHA."
                        ) from exc
                    if exc.code not in {408, 429, 500, 502, 503, 504}:
                        raise DownloadError(f"Public Download All trả HTTP {exc.code} cho {key}.") from exc
                    last_error = exc
                except DownloadError:
                    raise
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                if attempt + 1 < self.retries:
                    delay = min(2 ** attempt, 8)
                    LOGGER.warning("Public ZIP bị gián đoạn; giữ file .part và thử tiếp sau %ss (%s)", delay, key)
                    time.sleep(delay)
            else:
                raise DownloadError(f"Không tải được public ZIP cho {key} sau {self.retries} lần: {last_error}")
        return result

    def download(self, dataset_keys: list[str] | None = None) -> dict[str, list[Path]]:
        keys = dataset_keys or list(self.config["datasets"])
        result: dict[str, list[Path]] = {}
        for key in keys:
            if key not in self.config["datasets"]:
                raise ValueError(f"Dataset không có trong config: {key}")
            definition = self.config["datasets"][key]
            dataset_id = str(definition["dataset_id"])
            version = int(definition.get("version", 1))
            remote_files = self._list_files(dataset_id, version)
            include = list(definition.get("include_patterns", []))
            exclude = list(definition.get("exclude_patterns", []))
            if definition.get("download_mode") == "selective":
                selected = [item for item in remote_files if self._matches(item, include, exclude)]
            else:
                selected = [item for item in remote_files if self._matches(item, include, exclude)]
            if not selected:
                LOGGER.warning("Không có file khớp điều kiện tải cho %s; không tải gì.", key)
                result[key] = []
                continue
            root = project_path(self.config, f"data/raw/{key}/downloads")
            downloaded: list[Path] = []
            for item in selected:
                file_id = str(item.get("id") or "")
                filename = str(item.get("filename") or "")
                if not file_id or not filename:
                    LOGGER.warning("Bỏ qua metadata file thiếu id/filename: %r", item)
                    continue
                folder = str(item.get("_relative_folder", ""))
                rel = safe_relative_path(f"{folder}/{slugify(filename)}")
                # Prefixing with the immutable API file UUID avoids collisions
                # while preserving the original payload and its file suffix.
                target = root.joinpath(*rel.parts[:-1], f"{file_id}_{rel.parts[-1]}")
                target.parent.mkdir(parents=True, exist_ok=True)
                details = item.get("content_details") if isinstance(item.get("content_details"), dict) else {}
                expected_sha256 = str(details.get("sha256_hash") or "").lower()
                expected_size = int(details.get("size") or item.get("size") or 0)
                if target.exists() and target.stat().st_size > 0:
                    if expected_size and target.stat().st_size != expected_size:
                        raise DownloadError(
                            f"File đã tồn tại nhưng sai kích thước ({target.stat().st_size}/{expected_size}); "
                            "giữ nguyên raw để tránh ghi đè, hãy chuyển file nghi vấn ra ngoài rồi chạy lại."
                        )
                    if expected_sha256 and sha256_file(target) != expected_sha256:
                        raise DownloadError(
                            f"File đã tồn tại nhưng SHA256 không khớp; giữ nguyên raw: {target}"
                        )
                    LOGGER.info("Đã có, bỏ qua: %s", target)
                    downloaded.append(target)
                    continue
                url = self._url(
                    dataset_id,
                    f"files/{urllib.parse.quote(file_id)}/file_downloaded",
                    {"version": str(version)},
                )
                self._download_to(url, target, expected_sha256, expected_size)
                downloaded.append(target)
                LOGGER.info("Đã tải: %s", target)
            result[key] = downloaded
        return result

    def _download_to(self, url: str, target: Path, expected_sha256: str = "", expected_size: int = 0) -> None:
        partial = target.with_name(target.name + ".part")
        for attempt in range(self.retries):
            try:
                with self._open(url) as response, partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if partial.stat().st_size == 0:
                    raise DownloadError(f"File tải về rỗng: {url}")
                if expected_size and partial.stat().st_size != expected_size:
                    raise DownloadError(f"Kích thước tải về không khớp: {partial.stat().st_size}/{expected_size}")
                if expected_sha256 and sha256_file(partial) != expected_sha256:
                    raise DownloadError(f"SHA256 tải về không khớp: {target.name}")
                partial.replace(target)
                return
            except DownloadError:
                partial.unlink(missing_ok=True)
                raise
            except OSError as exc:
                partial.unlink(missing_ok=True)
                if attempt + 1 >= self.retries:
                    raise DownloadError(f"Lỗi ghi file {target}: {exc}") from exc
                time.sleep(min(2 ** attempt, 8))
