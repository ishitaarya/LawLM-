"""Optional GitHub Release backup/restore for LawLM training checkpoints.

Set LAWLM_GITHUB_TOKEN on the training machine. Checkpoints are stored as
release assets, not committed into the source repository.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_REPO = "ishitaarya/LawLM-"
DEFAULT_TAG = "qa-v2-checkpoints"
KEEP_REMOTE_CHECKPOINTS = 5


class GitHubCheckpointSync:
    def __init__(self, repo: str = DEFAULT_REPO, tag: str = DEFAULT_TAG):
        self.repo = os.environ.get("LAWLM_GITHUB_REPO", repo)
        self.tag = os.environ.get("LAWLM_GITHUB_RELEASE_TAG", tag)
        self.token = os.environ.get("LAWLM_GITHUB_TOKEN", "").strip()
        self.api_root = f"https://api.github.com/repos/{self.repo}"
        self.enabled = bool(self.token)

    def _request(self, url: str, method: str = "GET", data: bytes | None = None,
                 accept: str = "application/vnd.github+json",
                 content_type: str | None = None):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "LawLM-Checkpoint-Sync",
        }
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=180) as response:
            body = response.read()
            ctype = response.headers.get("Content-Type", "")
        if "json" in ctype or (body[:1] in (b"{", b"[")):
            return json.loads(body.decode("utf-8"))
        return body

    def _release(self, create: bool = False):
        if not self.enabled:
            return None
        url = f"{self.api_root}/releases/tags/{urllib.parse.quote(self.tag, safe='')}"
        try:
            return self._request(url)
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or not create:
                raise
        payload = json.dumps({
            "tag_name": self.tag,
            "name": "LawLM QA V2 Recovery Checkpoints",
            "body": "Automated rolling recovery checkpoints for QA V2 training. "
                    "These binary assets are not source-code files.",
            "draft": False,
            "prerelease": True,
        }).encode("utf-8")
        try:
            return self._request(f"{self.api_root}/releases", method="POST", data=payload)
        except urllib.error.HTTPError as exc:
            # If a concurrent run created the release, fetch it.
            if exc.code in (409, 422):
                return self._request(url)
            raise

    @staticmethod
    def _step(asset_name: str) -> int:
        match = re.search(r"resume_step_(\d+)\.pt$", asset_name)
        return int(match.group(1)) if match else -1

    def _assets(self, release: dict):
        return sorted(
            [a for a in release.get("assets", [])
             if self._step(a.get("name", "")) >= 0],
            key=lambda a: self._step(a["name"]),
            reverse=True,
        )

    def restore_latest_if_newer(self, directory: Path) -> Path | None:
        """Download remote newest checkpoint only if it is newer than local."""
        if not self.enabled:
            print("GitHub checkpoint sync disabled: set LAWLM_GITHUB_TOKEN to enable it.")
            return None
        try:
            release = self._release()
            if not release:
                print("No GitHub checkpoint release exists yet; local recovery will be used.")
                return None
            assets = self._assets(release)
            if not assets:
                print("GitHub checkpoint release has no recovery assets yet.")
                return None

            remote = assets[0]
            directory.mkdir(parents=True, exist_ok=True)
            local_files = list(directory.glob("lawsuit_llm_qa_v2_resume_step_*.pt"))
            local_latest = max(local_files, key=lambda p: self._step(p.name)) if local_files else None
            remote_step = self._step(remote["name"])
            if local_latest and self._step(local_latest.name) >= remote_step:
                print(f"Local recovery checkpoint is current (step {self._step(local_latest.name)}).")
                return local_latest

            asset_url = f"{self.api_root}/releases/assets/{remote['id']}"
            binary = self._request(asset_url, accept="application/octet-stream")
            if not isinstance(binary, bytes) or not binary:
                raise RuntimeError("GitHub did not return checkpoint binary data.")
            target = directory / remote["name"]
            temp = target.with_suffix(target.suffix + ".download")
            temp.write_bytes(binary)
            temp.replace(target)
            print(f"Downloaded GitHub recovery checkpoint: {target}")
            return target
        except Exception as exc:
            print(f"WARNING: GitHub checkpoint restore failed ({exc}). Trying local checkpoints.")
            return None

    def upload_and_prune(self, checkpoint_path: Path) -> bool:
        """Upload a checkpoint asset and keep only the newest five remote assets."""
        if not self.enabled:
            return False
        try:
            release = self._release(create=True)
            asset_name = checkpoint_path.name

            # Replacing the same step is safe (e.g. final save after periodic save).
            for asset in release.get("assets", []):
                if asset.get("name") == asset_name:
                    self._request(
                        f"{self.api_root}/releases/assets/{asset['id']}",
                        method="DELETE",
                    )

            upload_url = release["upload_url"].split("{", 1)[0]
            url = upload_url + "?" + urllib.parse.urlencode({"name": asset_name})
            # Upload one checkpoint at a time. Checkpoint files are expected to
            # remain below GitHub's per-release-asset size limit.
            binary = checkpoint_path.read_bytes()
            self._request(
                url,
                method="POST",
                data=binary,
                accept="application/vnd.github+json",
                content_type="application/octet-stream",
            )

            refreshed = self._release()
            assets = self._assets(refreshed)
            for asset in assets[KEEP_REMOTE_CHECKPOINTS:]:
                self._request(
                    f"{self.api_root}/releases/assets/{asset['id']}",
                    method="DELETE",
                )
            print(f"GitHub backup uploaded: {asset_name}; newest {KEEP_REMOTE_CHECKPOINTS} retained.")
            return True
        except Exception as exc:
            print(f"WARNING: GitHub checkpoint upload failed ({exc}). Local checkpoint is still saved. {exc}")
            return False
