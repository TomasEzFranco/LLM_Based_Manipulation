from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib import request, error

DEFAULT_PROMPT_PATH = str(
    Path(__file__).resolve().parents[1] / "prompts" / "live_sort_operator_v16.txt"
)
FINAL_JSON_START = "FINAL_JSON_START"
FINAL_JSON_END = "FINAL_JSON_END"

_ALLOWED_COMMANDS = {
    "observe_scene",
    "classify_cube",
    "grasp_cube",
    "push_cube",
    "pick_other",
    "return_cube",
    "return_placed_cube",
    "pick_placed_left",
    "pick_placed_right",
    "place_left",
    "place_right",
    "place_left_stack",
    "place_right_stack",
    "verify_last_place",
    "stop_run",
}


@dataclass
class LivePolicyConfig:
    backend: str = "ollama"  # ollama | none
    model_id: str = "gemma4:26b"
    endpoint: str = "http://127.0.0.1:11434"
    timeout_s: float = 2.5
    think: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    prompt_path: str = DEFAULT_PROMPT_PATH
    max_reprompt: int = 1
    num_predict: int = 0


@dataclass
class LivePolicyDecision:
    command: str
    reason: str
    confidence: float | None = None
    raw_output: str = ""
    valid: bool = True
    error: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    normalized_reason: str = ""
    normalized_from: str = ""


class LivePolicyBrain:
    """
    Small live policy adapter for promptable placement strategy.
    It never executes actions directly; it only returns one validated command.
    """

    def __init__(self, config: LivePolicyConfig):
        self.config = config
        self.prompt_text = Path(config.prompt_path).read_text(encoding="utf-8")

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            raise ValueError("empty model output")
        end_idx = text.rfind(FINAL_JSON_END)
        if end_idx < 0:
            raise ValueError("missing_final_json_block")
        start_idx = text.rfind(FINAL_JSON_START, 0, end_idx)
        if start_idx < 0:
            raise ValueError("missing_final_json_block")
        block = text[start_idx + len(FINAL_JSON_START) : end_idx].strip()
        if not block:
            raise ValueError("invalid_final_json_block")
        try:
            payload = json.loads(block)
        except Exception as exc:
            raise ValueError("invalid_final_json_block") from exc
        if not isinstance(payload, dict):
            raise ValueError("invalid_final_json_block")
        return payload

    @staticmethod
    def _validate_payload(
        payload: dict[str, Any],
        allowed_commands: set[str],
    ) -> tuple[str, str, float | None, str, str]:
        raw_command = str(payload.get("command", "")).strip()
        if not raw_command:
            raw_command = str(payload.get("action", "")).strip()
        if raw_command == "pick_misplaced_cube":
            # Enforce side-specific correction commands only.
            raise ValueError("invalid_generic_pick_misplaced")
        if raw_command == "pick_placed_cube":
            # Enforce side-specific correction commands only.
            raise ValueError("invalid_generic_pick_placed")
        normalized_reason = ""
        normalized_from = ""
        command = raw_command
        if command not in _ALLOWED_COMMANDS:
            raise ValueError(f"unsupported command '{command}'")
        if command not in allowed_commands:
            raise ValueError(f"command '{command}' not allowed now")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("missing reason")
        conf = payload.get("confidence", None)
        confidence = None
        if conf is not None:
            confidence = float(conf)
            confidence = max(0.0, min(1.0, confidence))
        return command, reason, confidence, normalized_reason, normalized_from

    def _call_ollama(self, prompt_input: str) -> tuple[str, float]:
        body = {
            "model": self.config.model_id,
            "prompt": f"{self.prompt_text}\n\nINPUT_JSON:\n{prompt_input}",
            "stream": False,
            "think": bool(self.config.think),
            "options": {
                "temperature": float(self.config.temperature),
                "top_p": float(self.config.top_p),
            },
        }
        num_predict = int(getattr(self.config, "num_predict", 0) or 0)
        if num_predict > 0:
            body["options"]["num_predict"] = int(num_predict)
        endpoint = str(self.config.endpoint).rstrip("/") + "/api/generate"
        req = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with request.urlopen(req, timeout=float(self.config.timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return str(payload.get("response", "")), dt_ms

    def decide(self, state: dict[str, Any], allowed_commands: list[str]) -> LivePolicyDecision:
        backend = str(self.config.backend).strip().lower()
        if backend in {"none", "off", "disabled"}:
            return LivePolicyDecision(
                command="observe_scene",
                reason="llm backend disabled",
                confidence=1.0,
                raw_output="",
                valid=False,
                error="backend_disabled",
                backend=backend,
                latency_ms=0.0,
            )

        last_error = ""
        last_raw = ""
        attempts = max(0, int(self.config.max_reprompt)) + 1
        input_payload = {
            "state": state,
            "allowed_commands": list(allowed_commands),
        }

        for attempt in range(1, attempts + 1):
            if attempt > 1 and last_error:
                input_payload["last_error"] = last_error
            prompt_input = json.dumps(input_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            try:
                if backend == "ollama":
                    raw, dt_ms = self._call_ollama(prompt_input)
                else:
                    raise ValueError(f"unsupported backend '{backend}'")
                last_raw = raw
                payload = self._extract_json(raw)
                cmd, reason, conf, normalized_reason, normalized_from = self._validate_payload(
                    payload, set(allowed_commands)
                )
                return LivePolicyDecision(
                    command=cmd,
                    reason=reason,
                    confidence=conf,
                    raw_output=raw,
                    valid=True,
                    error="",
                    backend=backend,
                    latency_ms=dt_ms,
                    normalized_reason=normalized_reason,
                    normalized_from=normalized_from,
                )
            except error.URLError as exc:
                last_error = f"network_error:{exc}"
            except Exception as exc:
                last_error = str(exc)
                if last_error == "invalid_generic_pick_misplaced":
                    print(
                        "[PolicyValidate] reason_code=invalid_generic_pick_misplaced "
                        "command=pick_misplaced_cube require_side_specific=true"
                    )
                if last_error == "invalid_generic_pick_placed":
                    print(
                        "[PolicyValidate] reason_code=invalid_generic_pick_placed "
                        "command=pick_placed_cube require_side_specific=true"
                    )

        return LivePolicyDecision(
            command="observe_scene",
            reason="llm output invalid after reprompt",
            confidence=None,
            raw_output=last_raw,
            valid=False,
            error=last_error or "invalid_output",
            backend=backend,
            latency_ms=0.0,
        )
