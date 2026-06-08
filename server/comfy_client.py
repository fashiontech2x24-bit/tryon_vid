"""Thin wrapper around the ComfyUI HTTP/WebSocket API.

ComfyUI exposes everything we need to drive inference headlessly:
  POST /upload/image     - put the reference image AND control video into input/
  POST /prompt           - queue a workflow (API format)
  WS   /ws?clientId=...   - live execution progress
  GET  /history/{id}     - finished job, including output filenames
  GET  /view             - download an output file
"""
import json
import os
import urllib.parse

import requests
import websocket  # websocket-client


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # -- health ---------------------------------------------------------------
    def is_up(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -- uploads --------------------------------------------------------------
    def upload_file(self, data: bytes, filename: str, content_type: str = "application/octet-stream") -> str:
        """Upload bytes into ComfyUI's input/ dir. Works for images and videos
        alike (VHS_LoadVideo reads from the same input dir). Returns the name
        ComfyUI stored it under (what you put in the node's widget)."""
        files = {"image": (filename, data, content_type)}
        form = {"type": "input", "overwrite": "true"}
        r = requests.post(f"{self.base_url}/upload/image", files=files, data=form, timeout=300)
        r.raise_for_status()
        body = r.json()
        name = body.get("name", filename)
        subfolder = body.get("subfolder", "")
        return os.path.join(subfolder, name) if subfolder else name

    # -- queue ----------------------------------------------------------------
    def queue_prompt(self, workflow: dict, client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        r = requests.post(f"{self.base_url}/prompt", json=payload, timeout=60)
        if r.status_code != 200:
            # ComfyUI returns a detailed validation error body
            raise RuntimeError(f"/prompt rejected ({r.status_code}): {r.text}")
        return r.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        return r.json().get(prompt_id, {})

    def get_file(self, filename: str, subfolder: str, file_type: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        url = f"{self.base_url}/view?" + urllib.parse.urlencode(params)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        return r.content

    # -- run + track ----------------------------------------------------------
    def run(self, workflow: dict, client_id: str, on_progress=None) -> dict:
        """Queue the workflow and block until it finishes, calling on_progress
        as execution advances. Returns the history entry for the prompt."""
        ws_url = (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + f"/ws?clientId={client_id}"
        )
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=30)
        try:
            prompt_id = self.queue_prompt(workflow, client_id)
            if on_progress:
                on_progress({"phase": "queued", "prompt_id": prompt_id})

            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue  # binary latent-preview frames; ignore
                evt = json.loads(msg)
                etype = evt.get("type")
                data = evt.get("data", {})

                if etype == "progress" and on_progress:
                    on_progress({
                        "phase": "sampling",
                        "value": data.get("value"),
                        "max": data.get("max"),
                    })
                elif etype == "executing":
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    if data.get("node") is None:
                        break  # this prompt is done
                    if on_progress:
                        on_progress({"phase": "executing", "node": data.get("node")})
                elif etype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise RuntimeError(f"ComfyUI execution error: {json.dumps(data)}")
                elif etype == "execution_interrupted" and data.get("prompt_id") == prompt_id:
                    raise RuntimeError("ComfyUI execution interrupted")
        finally:
            ws.close()

        return self.get_history(prompt_id)

    # -- output discovery -----------------------------------------------------
    @staticmethod
    def find_output_video(history: dict):
        """Scan a history entry's outputs for a saved video file.
        Returns (filename, subfolder, type) or None."""
        outputs = history.get("outputs", {})
        video_exts = (".mp4", ".webm", ".mov", ".mkv", ".gif")
        candidates = []
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for value in node_out.values():
                if not isinstance(value, list):
                    continue
                for item in value:
                    if isinstance(item, dict) and "filename" in item:
                        candidates.append(item)
        if not candidates:
            return None
        for item in candidates:
            if item.get("filename", "").lower().endswith(video_exts):
                return item["filename"], item.get("subfolder", ""), item.get("type", "output")
        first = candidates[0]
        return first["filename"], first.get("subfolder", ""), first.get("type", "output")
