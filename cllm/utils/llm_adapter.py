"""
utils/llm_adapter.py — CLLM v5 LLM 


----------
  "volc"     / Ark
              base_url : https://ark.cn-beijing.volces.com/api/v3
              model    : ep-xxxxxxxx  ID

  "apiyi"   APIYI  OpenAI 
              base_url : https://api.apiyi.com/v1
              model    : gpt-4.1-mini, gpt-4o, deepseek-chat 

  "openai"  OpenAI 
              base_url : https://api.openai.com/v1
              model    : gpt-4o, gpt-4.1-mini 

  "mock"    No LLM calls; returns placeholder data (for unit tests / mock mode)

How to switch platform
──────────────────────
Edit PLATFORM and the corresponding API_KEY / MODEL at the top of this file.
Runtime switching is not supported (hard-coded config ensures experiment reproducibility).
All calls go through the standard requests library to /chat/completions — no third-party SDK required.

Connectivity test
─────────────────
  python utils/llm_adapter.py
"""

import time
import json
import traceback
import requests
from typing import Optional
from datetime import datetime, timezone

# ============================================================
# ★ : 
# ============================================================

# Active platform. Options: "volc" | "apiyi" | "openai" | "mock"
PLATFORM = "volc"

# ── / Ark─────────────────────────────────────────────────────
VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
VOLC_API_KEY  = "sk-your-key-here"
VOLC_MODEL    = "ep-20260105123327-4k4rc"

# ── APIYI  ────────────────────────────────────────────────────────────
APIYI_BASE_URL = "https://api.apiyi.com/v1"
APIYI_API_KEY  = "sk-your-key-here"
APIYI_MODEL    = "gpt-4o"

# ── OpenAI  ───────────────────────────────────────────────────────────────
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_API_KEY  = "sk-your-openai-key-here"
OPENAI_MODEL    = "gpt-4o"

# ── General retry parameters ──────────────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_SLEEP = 30   # 

# ============================================================
# Internal: resolve actual parameters from PLATFORM (do not modify)
# ============================================================

_PLATFORM_MAP = {
    "volc":   (VOLC_BASE_URL,   VOLC_API_KEY,   VOLC_MODEL),
    "apiyi":  (APIYI_BASE_URL,  APIYI_API_KEY,  APIYI_MODEL),
    "openai": (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL),
    "mock":   ("",              "",              "mock"),
}

if PLATFORM not in _PLATFORM_MAP:
    raise ValueError(
        f"[llm_adapter] Unknown PLATFORM='{PLATFORM}'. "
        f"Valid options: {list(_PLATFORM_MAP.keys())}"
    )

_BASE_URL, _API_KEY, _MODEL = _PLATFORM_MAP[PLATFORM]

# LOG_FILE  main.py  outputs/  set_log_file() 
_LOG_FILE: Optional[str] = None


def set_log_file(path: str):
    """ main.py  LLM  outputs/ """
    global _LOG_FILE
    _LOG_FILE = path


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(text: str):
    if _LOG_FILE is None:
        return
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


# ──  ────────────────────────────────────────────────────────────────

class LLMFatalError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class LLMQuotaExhaustedError(LLMFatalError):
    """retryingresumable-run progress"""
    pass


_QUOTA_KEYWORDS = (
    "setlimitexceeded", "insufficient_quota", "quota_exceeded",
    "billingerror", "account_deactivated", "exceeded.*limit",
)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _QUOTA_KEYWORDS)


# ── LLMClient ────────────────────────────────────────────────────────────────

class LLMClient:

    def __init__(self):
        self._platform = PLATFORM
        self._base_url = _BASE_URL
        self._api_key  = _API_KEY
        self._model    = _MODEL
        if PLATFORM == "mock":
            print("[LLM] Backend: Mock LLM ")
        else:
            print(f"[LLM] Platform={PLATFORM}  Model={_MODEL}  BaseURL={_BASE_URL}")

    def invoke(self, prompt: str, system: Optional[str] = None) -> str:
        """
        
         /chat/completions  volc / apiyi / openai
        """
        _append_log(f"\n{'='*20} {_now_ts()} {'='*20}")
        _append_log(f"[platform] {self._platform}  [model] {self._model}")
        if system:
            _append_log(f"[system]\n{system}")
        _append_log(f"[prompt]\n{prompt}")

        # Mock 
        if self._platform == "mock":
            resp = json.dumps({"_mock": True, "result": "mock"})
            _append_log(f"[response]\n{resp}")
            return resp

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {"model": self._model, "messages": messages}

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]
                _append_log(f"[response]\n{result}")
                return result

            except Exception as e:
                last_error = e
                if _is_quota_error(e):
                    msg = f"[QUOTA EXHAUSTED] {type(e).__name__}: {e}"
                    print(f"[LLM] Quota exhausted — stopping. {e}")
                    _append_log(f"[QUOTA_EXHAUSTED] {msg}")
                    raise LLMQuotaExhaustedError(msg)

                print(f"[LLM Retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}")
                try:
                    print(f"  Response: {resp.text[:200]}")
                except Exception:
                    pass
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP)
                else:
                    traceback.print_exc()

        err = f"LLM failed after {MAX_RETRIES} retries: {last_error}"
        _append_log(f"[FATAL] {err}")
        raise LLMFatalError(err)

    def invoke_json(self, prompt: str, system: Optional[str] = None) -> dict:
        """ LLM  JSON """
        raw = self.invoke(prompt, system)
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0].strip()
                break
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s >= 0 and e > s:
                try:
                    return json.loads(raw[s:e])
                except Exception:
                    pass
            return {"_mock": True, "raw": raw[:200]}


# ── Singleton ─────────────────────────────────────────────────────────────────

_singleton: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton


# ── Connectivity testpython utils/llm_adapter.py─────────────────────────────────

def main_test():
    llm = get_llm()

    print("\n=== Test 1: Plain text invoke ===")
    answer = llm.invoke(
        prompt="What is 1 + 1? Reply with just the number.",
        system="You are a concise assistant.",
    )
    print("LLM response:", answer)

    print("\n=== Test 2: JSON invoke ===")
    result = llm.invoke_json(
        prompt='Return a JSON object: {"answer": <1+1>}. Only output the JSON.',
    )
    print("Parsed JSON:", result)

    if isinstance(result, dict) and result.get("_mock"):
        print(f"\n⚠  Got mock response — LLM NOT connected.")
        print(f"   PLATFORM='{PLATFORM}'  Check API key and network.")
    else:
        print(f"\n✓  LLM connected. Platform={PLATFORM}  Model={_MODEL}")


if __name__ == "__main__":
    main_test()
