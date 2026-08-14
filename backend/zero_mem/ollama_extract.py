"""
Ollama extractor — entity/relation extraction through an Ollama model
(cloud or local) instead of Gemini.

``gemini.py`` defines the shared SLM contract (schema, instruction,
SegmentExtraction, type canonicalisation); this module reuses it and adds the
one thing Ollama needs that Gemini does not: a **shape-tolerant parser**.

Ollama accepts a JSON-schema ``format`` parameter, but the hosted cloud models
treat it as a hint rather than a constraint. Measured against gemma4:31b-cloud
(2026-08-14) the model returns good Vietnamese extractions in the wrong shape:

    ```json                     <- markdown fence despite "return only JSON"
    [ { "segment": 0,           <- bare array, and 'segment' not 'index'
        "entities":[{"text":"Văn Tâm","label":"Person"}],   <- text/label
        "relations":[{"subject":"Văn Tâm","relation":"receives", ...}] } ]

The content is right; only the field names are wrong. Rejecting that would
throw away a perfectly good extraction, so the parser accepts the common
aliases and normalises them.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .extract import Mention, _canonical
from .gemini import (
    _INSTRUCTION,
    _JSON_SCHEMA,
    _PRONOUNS,
    MAX_CHARS_PER_CALL,
    MAX_SEGMENTS_PER_CALL,
    SegmentExtraction,
    canonicalize_type,
)

DEFAULT_OLLAMA_EXTRACT_MODEL = "gemma4:31b-cloud"
OLLAMA_TIMEOUT_S = 300.0

# Ollama models are chattier than Gemini about wrappers; say it plainly.
_OLLAMA_INSTRUCTION = _INSTRUCTION + (
    "\nReturn ONLY a JSON object of the form "
    '{"segments":[{"index":<int>,"entities":[{"name":...,"type":...}],'
    '"relations":[{"subject":...,"relation":...,"object":...}]}]}. '
    "No markdown fences, no commentary."
)

_NAME_KEYS = ("name", "text", "entity", "value", "surface")
_TYPE_KEYS = ("type", "label", "category", "kind")
_INDEX_KEYS = ("index", "segment", "segment_index", "id", "idx")
_REL_KEYS = ("relation", "predicate", "rel", "type")


class OllamaExtractorError(RuntimeError):
    pass


def _first(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _strip_fence(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _loads_lenient(text: str) -> Any:
    """Parse JSON, tolerating fences and trailing prose after the value."""
    cleaned = _strip_fence(text)
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    # Fall back to the first balanced {...} or [...] in the string.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == opener:
                depth += 1
            elif cleaned[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except ValueError:
                        break
    raise OllamaExtractorError("model did not return parseable JSON")


def normalize_payload(raw: str, count: int) -> Dict[int, SegmentExtraction]:
    """
    Turn whatever the model returned into {segment_index: SegmentExtraction}.

    Accepts a bare list or {"segments": [...]}, index under any of several
    names (falling back to positional order), and entity name/type under the
    aliases models actually emit.
    """
    data = _loads_lenient(raw)
    if isinstance(data, dict):
        items = data.get("segments")
        if items is None:
            # a single segment object, or {"0": {...}} style
            items = [data] if _first(data, _INDEX_KEYS) is not None else list(data.values())
    else:
        items = data
    if not isinstance(items, list):
        raise OllamaExtractorError("unexpected payload shape: %s" % type(items).__name__)

    out: Dict[int, SegmentExtraction] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_index = _first(item, _INDEX_KEYS)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = position          # model dropped the index; trust order
        if not (0 <= index < count):
            continue

        entities: Dict[str, Mention] = {}
        for e in item.get("entities") or []:
            if isinstance(e, str):
                name, etype = e.strip(), ""
            elif isinstance(e, dict):
                name = str(_first(e, _NAME_KEYS) or "").strip()
                etype = str(_first(e, _TYPE_KEYS) or "")
            else:
                continue
            if not name or len(name) > 80:
                continue
            key = _canonical(name)
            if not key:
                continue
            prev = entities.get(key)
            if prev is not None:
                entities[key] = Mention(prev.key, prev.name, prev.type, prev.count + 1)
            else:
                entities[key] = Mention(key, name, canonicalize_type(etype), 1)

        relations: List[Dict[str, str]] = []
        for r in item.get("relations") or []:
            if not isinstance(r, dict):
                continue
            subject = str(r.get("subject") or r.get("source") or "").strip()
            obj = str(r.get("object") or r.get("target") or "").strip()
            rel = str(_first(r, _REL_KEYS) or "").strip()
            if not (subject and obj and rel):
                continue
            if _canonical(subject) in _PRONOUNS or _canonical(obj) in _PRONOUNS:
                continue
            # models emit "is_relic_of" / "receives"; normalise to the same
            # SHORT_UPPERCASE_VERB shape Gemini produces
            rel = re.sub(r"\s+", "_", rel).upper()[:40]
            relations.append({"subject": subject, "relation": rel, "object": obj})

        out[index] = SegmentExtraction(list(entities.values()), relations)
    return out


class OllamaExtractor(object):
    """Same interface as GeminiExtractor, backed by an Ollama model."""

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EXTRACT_MODEL,
        base_url: str = "http://localhost:11434",
        client: Optional[httpx.Client] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=OLLAMA_TIMEOUT_S)

    def close(self) -> None:
        self._client.close()

    def extract_segments(self, texts: Sequence[str]) -> List[SegmentExtraction]:
        texts = list(texts)
        out: List[SegmentExtraction] = [SegmentExtraction([], []) for _ in texts]
        batch: List[int] = []
        chars = 0
        for i, text in enumerate(texts):
            if batch and (len(batch) >= MAX_SEGMENTS_PER_CALL or chars + len(text) > MAX_CHARS_PER_CALL):
                self._run(texts, batch, out)
                batch, chars = [], 0
            batch.append(i)
            chars += len(text)
        if batch:
            self._run(texts, batch, out)
        return out

    def _run(self, texts: Sequence[str], indices: List[int], out: List[SegmentExtraction]) -> None:
        # Number segments from 0 within the batch so the model's positional
        # fallback lines up even when it omits indices entirely.
        numbered = "\n\n".join("[%d]\n%s" % (n, texts[i]) for n, i in enumerate(indices))
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _OLLAMA_INSTRUCTION},
                {"role": "user", "content": numbered},
            ],
            "stream": False,
            "format": _JSON_SCHEMA,
            "options": {"temperature": 0},
        }
        try:
            resp = self._client.post(self.base_url + "/api/chat", json=body)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content")
        except httpx.HTTPError as exc:
            raise OllamaExtractorError("Ollama API error: %s" % exc)
        if not isinstance(content, str) or not content.strip():
            raise OllamaExtractorError("Ollama returned an empty response")

        parsed = normalize_payload(content, len(indices))
        for local_index, seg_index in enumerate(indices):
            if local_index in parsed:
                out[seg_index] = parsed[local_index]


def create_ollama_extractor(
    model: str = DEFAULT_OLLAMA_EXTRACT_MODEL,
    base_url: str = "http://localhost:11434",
    logger=None,
) -> Optional[OllamaExtractor]:
    try:
        return OllamaExtractor(model, base_url)
    except Exception as exc:  # pragma: no cover
        if logger:
            logger("zero-mem: Ollama extractor unavailable (%s); using local NER." % exc)
        return None
