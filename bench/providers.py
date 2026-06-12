"""Provider adapters for the SemiHS-Bench runner.

Adapters normalize raw provider responses into a small common envelope while
preserving provider-visible payloads for local trace artifacts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class ProviderRequest:
    provider: str
    model_id: str
    system: str
    user: str
    mode: str
    tier: int
    frozen_id: str
    reasoning_level: Optional[str] = None
    web_enabled: bool = False
    temperature: Optional[float] = 0.0
    max_output_tokens: int = 10000
    timeout_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    private: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    text: str
    raw: Any = None
    request_payload: Any = None
    usage: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    thought_summaries: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class ProviderError(RuntimeError):
    """Raised when a provider cannot complete a request."""


class BaseProvider:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.name = str(config.get("name") or config.get("model") or config.get("provider"))
        self.model_id = str(config.get("model") or config.get("model_id") or self.name)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK objects into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    for method_name in ("model_dump", "dict", "to_dict", "to_json_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                if method_name == "model_dump":
                    return to_jsonable(method(mode="json"))
                return to_jsonable(method())
            except TypeError:
                try:
                    return to_jsonable(method())
                except Exception:
                    pass
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return repr(value)


def _env_value(config: Mapping[str, Any], key_name: str, default_env: str) -> Optional[str]:
    env_name = str(config.get(key_name) or default_env)
    return os.environ.get(env_name)


def _text_from_openai_response(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    payload = to_jsonable(resp)
    chunks: List[str] = []
    for item in payload.get("output", []) if isinstance(payload, dict) else []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "".join(chunks)


def _openai_usage(resp: Any) -> Dict[str, Any]:
    payload = to_jsonable(resp)
    return dict(payload.get("usage") or {}) if isinstance(payload, dict) else {}


def _openai_tool_calls(resp: Any) -> List[Dict[str, Any]]:
    payload = to_jsonable(resp)
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out
    for item in payload.get("output", []) or []:
        item_type = item.get("type")
        if item_type and item_type != "message":
            out.append(item)
    return out


def _openai_citations(resp: Any) -> List[Dict[str, Any]]:
    payload = to_jsonable(resp)
    citations: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return citations
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                citations.append(annotation)
    return citations


def _openai_thought_summaries(resp: Any) -> List[str]:
    payload = to_jsonable(resp)
    summaries: List[str] = []
    if not isinstance(payload, dict):
        return summaries
    for item in payload.get("output", []) or []:
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary") or []
        if isinstance(summary, str):
            summaries.append(summary)
            continue
        for part in summary:
            if isinstance(part, str):
                summaries.append(part)
            elif isinstance(part, Mapping) and part.get("text"):
                summaries.append(str(part["text"]))
    return summaries


def _openai_reasoning_temperature_model(model_id: str, reasoning_level: Optional[str]) -> bool:
    model = model_id.lower()
    return bool(reasoning_level) or model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIProvider(BaseProvider):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProviderError("Install the 'openai' package to use provider=openai") from exc
        kwargs: Dict[str, Any] = {}
        api_key = _env_value(config, "api_key_env", "OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if config.get("base_url"):
            kwargs["base_url"] = str(config["base_url"])
        if config.get("timeout_s"):
            kwargs["timeout"] = float(config["timeout_s"])
        self.client = OpenAI(**kwargs)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        warnings: List[str] = []
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "instructions": request.system,
            "input": request.user,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_level:
            reasoning: Dict[str, Any] = {"effort": request.reasoning_level}
            if request.reasoning_level != "none" and self.config.get("include_reasoning_summary", True):
                reasoning["summary"] = str(self.config.get("reasoning_summary") or "auto")
            payload["reasoning"] = reasoning
            payload["include"] = ["reasoning.encrypted_content"]
        if request.web_enabled:
            payload.setdefault("include", [])
            if "web_search_call.action.sources" not in payload["include"]:
                payload["include"].append("web_search_call.action.sources")
            tool: Dict[str, Any] = {"type": "web_search"}
            if request.extra.get("search_context_size"):
                tool["search_context_size"] = request.extra["search_context_size"]
            payload["tools"] = [tool]
            payload["tool_choice"] = "auto"
        payload.update(dict(self.config.get("request_overrides") or {}))
        if _openai_reasoning_temperature_model(request.model_id, request.reasoning_level) and payload.get("temperature") != 1.0:
            if "temperature" in payload:
                warnings.append("openai_temperature_forced_to_1_for_reasoning_model")
            payload["temperature"] = 1.0
        started = time.time()
        resp = self.client.responses.create(**payload)
        return ProviderResponse(
            text=_text_from_openai_response(resp),
            raw=to_jsonable(resp),
            request_payload=payload,
            usage=_openai_usage(resp),
            citations=_openai_citations(resp),
            tool_calls=_openai_tool_calls(resp),
            thought_summaries=_openai_thought_summaries(resp),
            warnings=warnings,
            elapsed_s=time.time() - started,
        )


def _anthropic_content(resp: Any) -> tuple:
    payload = to_jsonable(resp)
    texts: List[str] = []
    thoughts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for block in payload.get("content", []) or []:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                texts.append(str(block["text"]))
                citations.extend(block.get("citations", []) or [])
            elif btype == "thinking" and block.get("thinking"):
                thoughts.append(str(block["thinking"]))
            elif btype in {"server_tool_use", "web_search_tool_result", "tool_use"}:
                tool_calls.append(block)
    return "".join(texts), thoughts, citations, tool_calls


class AnthropicProvider(BaseProvider):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ProviderError("Install the 'anthropic' package to use provider=anthropic") from exc
        kwargs: Dict[str, Any] = {}
        api_key = _env_value(config, "api_key_env", "ANTHROPIC_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if config.get("base_url"):
            kwargs["base_url"] = str(config["base_url"])
        if config.get("timeout_s"):
            kwargs["timeout"] = float(config["timeout_s"])
        self.client = anthropic.Anthropic(**kwargs)

    @staticmethod
    def _temperature_deprecated(model_id: str) -> bool:
        """True for Anthropic models that reject the temperature parameter entirely."""
        m = model_id.lower()
        return "opus-4" in m

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        warnings: List[str] = []
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "max_tokens": request.max_output_tokens,
        }
        if request.temperature is not None and not self._temperature_deprecated(request.model_id):
            payload["temperature"] = request.temperature
        if request.reasoning_level == "none":
            payload["thinking"] = {"type": "disabled"}
        elif request.reasoning_level:
            payload["output_config"] = {"effort": request.reasoning_level}
            if self.config.get("thinking", "adaptive") != "disabled":
                payload["thinking"] = {"type": "adaptive", "display": "summarized"}
        if request.web_enabled:
            payload["tools"] = [{
                "type": str(self.config.get("web_search_type") or "web_search_20250305"),
                "name": "web_search",
                "max_uses": int(self.config.get("web_max_uses") or request.extra.get("web_max_uses") or 5),
            }]
        payload.update(dict(self.config.get("request_overrides") or {}))
        if request.reasoning_level and request.reasoning_level != "none" and not self._temperature_deprecated(request.model_id):
            if payload.get("temperature") != 1.0:
                if "temperature" in payload:
                    warnings.append("anthropic_temperature_forced_to_1_for_reasoning")
                payload["temperature"] = 1.0
        elif self._temperature_deprecated(request.model_id):
            payload.pop("temperature", None)
            warnings.append("anthropic_temperature_omitted_deprecated_for_model")
        started = time.time()
        resp = self.client.messages.create(**payload)
        text, thoughts, citations, tool_calls = _anthropic_content(resp)
        raw = to_jsonable(resp)
        return ProviderResponse(
            text=text,
            raw=raw,
            request_payload=payload,
            usage=dict(raw.get("usage") or {}) if isinstance(raw, dict) else {},
            citations=citations,
            tool_calls=tool_calls,
            thought_summaries=thoughts,
            warnings=warnings,
            elapsed_s=time.time() - started,
        )


def _gemini_text_and_thoughts(resp: Any) -> tuple:
    text = getattr(resp, "text", None)
    if text:
        answer = str(text)
    else:
        answer = ""
    thoughts: List[str] = []
    payload = to_jsonable(resp)
    if isinstance(payload, dict):
        for cand in payload.get("candidates", []) or []:
            content = cand.get("content") or {}
            for part in content.get("parts", []) or []:
                if part.get("thought") and part.get("text"):
                    thoughts.append(str(part["text"]))
                elif not answer and part.get("text"):
                    answer += str(part["text"])
    return answer, thoughts


def _gemini_grounding(resp: Any) -> List[Dict[str, Any]]:
    payload = to_jsonable(resp)
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out
    for cand in payload.get("candidates", []) or []:
        grounding = cand.get("grounding_metadata") or cand.get("groundingMetadata")
        if grounding:
            out.append(grounding)
    return out


def _gemini_finish_reasons(raw: Any) -> List[str]:
    if not isinstance(raw, dict):
        return []
    out: List[str] = []
    for cand in raw.get("candidates", []) or []:
        reason = cand.get("finish_reason") or cand.get("finishReason")
        if reason:
            out.append(str(reason))
    return out


class TavilySearchClient:
    """Minimal Tavily search client using stdlib urllib - no extra deps."""

    _API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        chunks_per_source: int = 3,
    ) -> None:
        self.api_key = api_key
        self.search_depth = search_depth
        self.include_raw_content = include_raw_content
        self.chunks_per_source = chunks_per_source

    def search(self, query: str, max_results: int = 5, timeout_s: float = 20.0) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth,
        }
        if self.include_raw_content:
            body["include_raw_content"] = "markdown"
            body["chunks_per_source"] = self.chunks_per_source
        payload = json.dumps(body).encode()
        req = urllib.request.Request(self._API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Tavily HTTP {exc.code}: {detail[:300]}") from exc
        except Exception as exc:
            raise ProviderError(f"Tavily search failed: {exc}") from exc
        return list(data.get("results") or [])


def _tavily_context_block(results: List[Dict[str, Any]], max_content_chars: int = 800) -> str:
    lines = ["[WEB SEARCH CONTEXT]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '').strip()}")
        if r.get("url"):
            lines.append(f"   Source: {r['url']}")
        # prefer raw_content (markdown) if present, fall back to content summary
        content = str(r.get("raw_content") or r.get("content") or "").strip()
        if content:
            lines.append(f"   {content[:max_content_chars]}")
        lines.append("")
    lines.append("[END WEB SEARCH CONTEXT]")
    return "\n".join(lines)


class GeminiProvider(BaseProvider):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise ProviderError("Install the 'google-genai' package to use provider=gemini") from exc
        self.types = types
        kwargs: Dict[str, Any] = {}
        api_key = _env_value(config, "api_key_env", "GEMINI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        self.client = genai.Client(**kwargs)
        self.tavily: Optional[TavilySearchClient] = None
        if config.get("tavily_fallback"):
            tavily_key = os.environ.get("TAVILY_API_KEY")
            if tavily_key:
                self.tavily = TavilySearchClient(
                    api_key=tavily_key,
                    search_depth=str(config.get("tavily_search_depth") or "basic"),
                    include_raw_content=bool(config.get("tavily_include_raw_content", False)),
                    chunks_per_source=int(config.get("tavily_chunks_per_source") or 3),
                )
            else:
                import warnings as _warnings
                _warnings.warn("tavily_fallback=true but TAVILY_API_KEY is not set; falling back to native Google search")

    def _tavily_tool(self) -> Any:
        return self.types.Tool(
            function_declarations=[
                self.types.FunctionDeclaration(
                    name="tavily_web_search",
                    description=(
                        "Search the web for product specifications, HS tariff codes, "
                        "trade classification information, and customs data."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "Search query, e.g. the product name plus 'HS code classification'.",
                            }
                        },
                        "required": ["query"],
                    },
                )
            ]
        )

    def _run_with_tavily_tools(
        self,
        request: ProviderRequest,
        gen_config: Any,
        warnings: List[str],
    ) -> tuple:
        """Multi-turn Gemini call where Gemini invokes Tavily as a function tool."""
        max_calls = int(self.config.get("web_max_uses") or request.extra.get("web_max_uses") or 5)
        max_results = int(self.config.get("tavily_max_results") or 5)
        contents: Any = request.user
        resp = None
        accumulated_usage: Dict[str, Any] = {}
        tool_call_count = 0

        for _ in range(max_calls + 1):
            resp = self.client.models.generate_content(
                model=request.model_id,
                contents=contents,
                config=gen_config,
            )
            # Accumulate usage across turns
            raw_turn = to_jsonable(resp)
            turn_usage = dict(raw_turn.get("usage_metadata") or raw_turn.get("usageMetadata") or {}) if isinstance(raw_turn, dict) else {}
            for k, v in turn_usage.items():
                if isinstance(v, (int, float)):
                    accumulated_usage[k] = accumulated_usage.get(k, 0) + v

            # Collect function calls from this response
            fc_list: List[Any] = []
            for cand in (resp.candidates or []):
                for part in (getattr(cand.content, "parts", None) or []):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        fc_list.append(fc)

            if not fc_list:
                break  # Final answer - no more tool calls

            # Execute each Tavily search call
            response_parts: List[Any] = []
            for fc in fc_list:
                name = str(fc.name or "")
                args = dict(fc.args or {})
                query = str(args.get("query") or "").strip()
                if name == "tavily_web_search" and query:
                    try:
                        results = self.tavily.search(query, max_results=max_results)
                        result_text = _tavily_context_block(results) if results else "No results found."
                        tool_call_count += 1
                        warnings.append(f"gemini_tavily_tool_call_{tool_call_count}")
                    except ProviderError as exc:
                        result_text = f"Search error: {exc}"
                        warnings.append("gemini_tavily_tool_error")
                else:
                    result_text = "Unknown tool or missing query."
                response_parts.append(
                    self.types.Part(
                        function_response=self.types.FunctionResponse(
                            name=name,
                            response={"result": result_text},
                        )
                    )
                )

            # Build multi-turn contents for next iteration
            if isinstance(contents, str):
                user_turn = self.types.Content(role="user", parts=[self.types.Part(text=contents)])
                contents = [user_turn, resp.candidates[0].content, self.types.Content(role="user", parts=response_parts)]
            else:
                contents = list(contents) + [resp.candidates[0].content, self.types.Content(role="user", parts=response_parts)]

        return resp, accumulated_usage

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        warnings: List[str] = []
        thinking_kwargs: Dict[str, Any] = {}
        if request.reasoning_level:
            thinking_kwargs["thinking_level"] = request.reasoning_level
        include_thoughts = bool(self.config.get("include_thoughts", True))
        if include_thoughts:
            thinking_kwargs["include_thoughts"] = True
        config_kwargs: Dict[str, Any] = {
            "system_instruction": request.system,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if thinking_kwargs:
            config_kwargs["thinking_config"] = self.types.ThinkingConfig(**thinking_kwargs)
        if request.web_enabled:
            if self.tavily:
                config_kwargs["tools"] = [self._tavily_tool()]
            else:
                config_kwargs["tools"] = [self.types.Tool(google_search=self.types.GoogleSearch())]
        config_kwargs.update(dict(self.config.get("request_overrides") or {}))
        gen_config = self.types.GenerateContentConfig(**config_kwargs)
        request_payload = {
            "model": request.model_id,
            "contents": request.user,
            "config": to_jsonable(gen_config),
        }
        started = time.time()
        if request.web_enabled and self.tavily:
            resp, usage = self._run_with_tavily_tools(request, gen_config, warnings)
        else:
            resp = self.client.models.generate_content(
                model=request.model_id,
                contents=request.user,
                config=gen_config,
            )
            raw_u = to_jsonable(resp)
            usage = dict(raw_u.get("usage_metadata") or raw_u.get("usageMetadata") or {}) if isinstance(raw_u, dict) else {}
        text, thoughts = _gemini_text_and_thoughts(resp)
        raw = to_jsonable(resp)
        finish_reasons = _gemini_finish_reasons(raw)
        warnings += [f"gemini_finish_reason_{reason}" for reason in finish_reasons if reason != "STOP"]
        if not text and any(reason in {"MALFORMED_FUNCTION_CALL"} for reason in finish_reasons):
            raise ProviderError(f"Gemini returned no text; finish_reason={','.join(finish_reasons)}")
        if not text:
            warnings.append("gemini_empty_response")
        thoughts_token_count = usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount")
        if thoughts_token_count and not thoughts:
            if include_thoughts:
                warnings.append("gemini_thought_tokens_without_returned_summary")
            else:
                warnings.append("gemini_thought_summaries_not_requested")
        return ProviderResponse(
            text=text,
            raw=raw,
            request_payload=request_payload,
            usage=usage,
            citations=_gemini_grounding(resp),
            tool_calls=_gemini_grounding(resp),
            thought_summaries=thoughts,
            warnings=warnings,
            elapsed_s=time.time() - started,
        )


class OpenAICompatibleProvider(BaseProvider):
    """Chat Completions adapter for vLLM/SGLang/Ollama/LM Studio/Together-style APIs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProviderError("Install the 'openai' package to use provider=openai_compatible") from exc
        kwargs: Dict[str, Any] = {"base_url": str(config.get("base_url") or os.environ.get("OPENAI_COMPAT_BASE_URL") or "")}
        if not kwargs["base_url"]:
            raise ProviderError("provider=openai_compatible requires base_url or OPENAI_COMPAT_BASE_URL")
        kwargs["api_key"] = _env_value(config, "api_key_env", "OPENAI_COMPAT_API_KEY") or "not-needed"
        if config.get("timeout_s"):
            kwargs["timeout"] = float(config["timeout_s"])
        self.client = OpenAI(**kwargs)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        warnings: List[str] = []
        extra_body = dict(self.config.get("extra_body") or {})
        if request.reasoning_level and "reasoning_effort" not in extra_body:
            extra_body["reasoning_effort"] = request.reasoning_level
        if request.web_enabled and not self.config.get("tools"):
            warnings.append("web_enabled_requested_but_openai_compatible_adapter_has_no_native_web_tool")
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if extra_body:
            payload["extra_body"] = extra_body
        if self.config.get("tools"):
            payload["tools"] = self.config["tools"]
        payload.update(dict(self.config.get("request_overrides") or {}))
        started = time.time()
        resp = self.client.chat.completions.create(**payload)
        raw = to_jsonable(resp)
        text = ""
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices:
                text = str(((choices[0].get("message") or {}).get("content")) or "")
        return ProviderResponse(
            text=text,
            raw=raw,
            request_payload=payload,
            usage=dict(raw.get("usage") or {}) if isinstance(raw, dict) else {},
            warnings=warnings,
            elapsed_s=time.time() - started,
        )


def _openrouter_citations(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for choice in (raw.get("choices") or []):
        for ann in ((choice.get("message") or {}).get("annotations") or []):
            if ann.get("type") == "url_citation":
                citations.append(ann.get("url_citation") or ann)
    return citations


class OpenRouterProvider(BaseProvider):
    """OpenRouter chat completions adapter with openrouter:web_search server tool support."""

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ProviderError("Install the 'openai' package to use provider=openrouter") from exc
        api_key = _env_value(config, "api_key_env", "OPENROUTER_API_KEY") or "not-needed"
        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": str(config.get("base_url") or self._BASE_URL),
            "default_headers": {
                "HTTP-Referer": "https://github.com/cti-iel/SemiHS-Bench",
                "X-Title": "SemiHS-Bench",
            },
        }
        if config.get("timeout_s"):
            kwargs["timeout"] = float(config["timeout_s"])
        self.client = OpenAI(**kwargs)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        warnings: List[str] = []
        payload: Dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        # All OpenRouter-specific params go in extra_body - the OpenAI SDK rejects unknown kwargs.
        # provider.allow_fallbacks is true by default in OpenRouter - no need to set it explicitly.
        # fallback_models: ["other/model", ...] → OpenRouter tries primary first, then the list.
        extra_body: Dict[str, Any] = dict(self.config.get("extra_body") or {})
        if request.reasoning_level is not None:
            extra_body["reasoning"] = {"effort": request.reasoning_level}
        if request.web_enabled:
            web_max_uses = int(self.config.get("web_max_uses") or request.extra.get("web_max_uses") or 5)
            search_params: Dict[str, Any] = {"max_total_results": web_max_uses * 5}
            ctx_size = request.extra.get("search_context_size")
            if ctx_size:
                search_params["search_context_size"] = ctx_size
            extra_body["tools"] = [{"type": "openrouter:web_search", "parameters": search_params}]
        fallback_models = list(self.config.get("fallback_models") or [])
        if fallback_models:
            extra_body["models"] = [request.model_id] + fallback_models
        if extra_body:
            payload["extra_body"] = extra_body
        payload.update(dict(self.config.get("request_overrides") or {}))
        started = time.time()
        resp = self.client.chat.completions.create(**payload)
        raw = to_jsonable(resp)
        text = ""
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices:
                text = str(((choices[0].get("message") or {}).get("content")) or "")
                finish = choices[0].get("finish_reason") or ""
                if finish and finish != "stop":
                    warnings.append(f"openrouter_finish_reason_{finish}")
        usage = dict(raw.get("usage") or {}) if isinstance(raw, dict) else {}
        stu = usage.pop("server_tool_use", None) or {}
        if stu:
            usage["web_search_requests"] = stu.get("web_search_requests", 0)
        return ProviderResponse(
            text=text,
            raw=raw,
            request_payload=payload,
            usage=usage,
            citations=_openrouter_citations(raw) if isinstance(raw, dict) else [],
            warnings=warnings,
            elapsed_s=time.time() - started,
        )


def _normalize_hs6(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())[:6]


def _collect_hs6_codes(value: Any) -> List[str]:
    codes: List[str] = []
    seen = set()

    def add(raw: Any) -> None:
        code = _normalize_hs6(raw)
        if len(code) == 6 and code not in seen:
            seen.add(code)
            codes.append(code)

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key in ("hscode", "hs_code", "code", "hscode_number", "top_hscode"):
                if key in node:
                    add(node[key])
            for nested in node.values():
                walk(nested)
            return
        if isinstance(node, list):
            for nested in node:
                walk(nested)
            return
        add(node)

    walk(value)
    return codes


class HSCodeApiProvider(BaseProvider):
    """Adapter for the local HSCode API service."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.base_url = str(
            config.get("base_url")
            or os.environ.get(str(config.get("base_url_env") or "HSCODE_API_BASE_URL"))
            or "http://127.0.0.1:8005"
        ).rstrip("/")
        self.endpoint = str(config.get("endpoint") or "/v1/classify/resolve")
        self.enable_hs6_verifier = bool(config.get("enable_hs6_verifier", True))
        self.include_toc = bool(config.get("include_TOC", True))
        self.number_of_probable_chapters = str(config.get("number_of_probable_chapters") or "7")

    def _payload_from_record(self, request: ProviderRequest) -> Dict[str, Any]:
        record = dict(request.private.get("record") or {})
        tier_minimal = record.get("tier2_minimal") if isinstance(record.get("tier2_minimal"), Mapping) else {}

        if request.tier == 2:
            part_name = str(tier_minimal.get("part_name") or record.get("tier1_description") or request.frozen_id)
            manufacturer = str(tier_minimal.get("manufacturer") or "unknown")
            product_description = part_name
            extra_details = None
        else:
            product_description = str(record.get("tier1_description") or request.user)
            part_name = product_description[:160] or request.frozen_id
            source_meta = record.get("source_metadata") if isinstance(record.get("source_metadata"), Mapping) else {}
            manufacturer = str(source_meta.get("manufacturer_hint") or "unknown")
            extra_details = None

        payload: Dict[str, Any] = {
            "request_id": request.frozen_id,
            "part_name": part_name,
            "manufacturer": manufacturer,
            "product_description": product_description,
            "enrich_description": False,
            "number_of_probable_chapters": self.number_of_probable_chapters,
            "include_TOC": self.include_toc,
            "enable_hs6_verifier": self.enable_hs6_verifier,
        }
        if extra_details:
            payload["extra_details"] = extra_details
        return payload

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        payload = self._payload_from_record(request)
        data = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url=f"{self.base_url}{self.endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout_s or 600) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"HSCode API HTTP {exc.code}: {detail[:500]}") from exc

        result = ((raw.get("hscode_result") or {}).get("result") or {}) if isinstance(raw, dict) else {}
        service_codes = _collect_hs6_codes([
            result.get("top_hscode"),
            result.get("verified_hscode_ranking"),
            result.get("probable_hscode_list"),
        ])
        if request.mode == "constrained":
            candidates = [str(code) for code in request.private.get("candidate_codes") or []]
            candidate_set = {_normalize_hs6(code): code for code in candidates}
            ranked = [candidate_set[code] for code in service_codes if code in candidate_set]
            ranked.extend(code for code in candidates if code not in ranked)
        else:
            ranked = service_codes or ["000000"]

        return ProviderResponse(
            text=json.dumps(ranked),
            raw=raw,
            request_payload=payload,
            elapsed_s=time.time() - started,
        )


class MockProvider(BaseProvider):
    """Deterministic provider for smoke tests and CI."""

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        strategy = str(self.config.get("strategy") or "first")
        delay_s = float(self.config.get("delay_s") or 0)
        started = time.time()
        if delay_s > 0:
            time.sleep(delay_s)
        if request.mode == "constrained":
            candidate_codes: Sequence[str] = request.private.get("candidate_codes") or []
            if strategy == "gold" and request.private.get("gold_code") in candidate_codes:
                gold_idx = list(candidate_codes).index(request.private["gold_code"])
                indices = [gold_idx] + [i for i in range(len(candidate_codes)) if i != gold_idx]
            else:
                indices = list(range(len(candidate_codes)))
            text = json.dumps(indices)
        else:
            if strategy == "gold":
                codes = [str(request.private.get("gold_code") or "000000")]
            else:
                codes = list(request.private.get("open_codes") or ["000000"])[:5]
            text = json.dumps(codes)
        raw = {
            "mock": True,
            "strategy": strategy,
            "text": text,
            "model": request.model_id,
            "reasoning_level": request.reasoning_level,
            "web_enabled": request.web_enabled,
        }
        return ProviderResponse(
            text=text,
            raw=raw,
            request_payload={
                "model": request.model_id,
                "system": request.system,
                "user": request.user,
            },
            usage={"input_tokens": len(request.system.split()) + len(request.user.split()), "output_tokens": len(text.split())},
            elapsed_s=time.time() - started,
        )


def build_provider(config: Mapping[str, Any]) -> BaseProvider:
    provider = str(config.get("provider") or "").lower()
    if provider == "openai":
        return OpenAIProvider(config)
    if provider == "anthropic":
        return AnthropicProvider(config)
    if provider == "gemini":
        return GeminiProvider(config)
    if provider == "openrouter":
        return OpenRouterProvider(config)
    if provider in {"openai_compatible", "oss", "compatible"}:
        return OpenAICompatibleProvider(config)
    if provider in {"hscode_api", "hscode-api", "hscode"}:
        return HSCodeApiProvider(config)
    if provider in {"mock", "mock_provider"}:
        return MockProvider(config)
    raise ProviderError(f"Unknown provider {provider!r}")
