"""
Gemini SLM extractor (Google AI Studio) for Zero-Mem.

Adds what the token-free gazetteer/pattern path cannot infer: relation triples
``(subject) -[RELATION]-> (object)`` and typed entities for names the context
files haven't declared yet. Default model is ``gemini-2.5-flash-lite`` — the
cheapest structured-output model on the API, with thinking off by default, so
extraction spends almost nothing.

Design constraints (learned from the Bun MCP implementation of the same idea):

* **Batch per document.** One request extracts every segment of a chapter at
  once; per-segment calls would mean ~25 round-trips per ingest.
* **Never fail an ingest.** Any API error is reported to the caller as "no API
  result"; the deterministic local pipeline always runs regardless.
* **Version-tolerant request shape.** The structured-output field names differ
  between Gemini generations (2.5 uses ``responseSchema`` with an UPPERCASE
  Type enum; 3.x uses ``response_format`` with lowercase JSON Schema), so the
  extractor tries the shape matching its model first and falls back on a 400.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .extract import CONCEPT, ITEM, Mention, NUMBER, PERSON, PLACE, PROPER, _canonical

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_EXTRACT_MODEL = "gemini-2.5-flash-lite"
API_TIMEOUT_S = 45.0
MAX_SEGMENTS_PER_CALL = 40
MAX_CHARS_PER_CALL = 30_000


class SegmentExtraction(object):
    __slots__ = ("entities", "relations")

    def __init__(self, entities: List[Mention], relations: List[Dict[str, str]]):
        self.entities = entities
        self.relations = relations

    def to_json(self) -> str:
        return json.dumps({
            "entities": [
                {"key": m.key, "name": m.name, "type": m.type, "count": m.count}
                for m in self.entities
            ],
            "relations": self.relations,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SegmentExtraction":
        data = json.loads(payload)
        entities = [
            Mention(e["key"], e["name"], e["type"], int(e.get("count", 1)))
            for e in data.get("entities", [])
        ]
        return cls(entities, list(data.get("relations", [])))


def canonicalize_type(raw: str) -> str:
    """Map the SLM's free-form labels onto this port's fixed taxonomy."""
    t = (raw or "").strip().upper()
    if t.startswith(("PERSON", "PEOPLE", "CHARACTER", "NHÂN")):
        return PERSON
    if t.startswith(("LOCATION", "PLACE", "GPE", "CITY", "COUNTRY", "SETTING", "ĐỊA")):
        return PLACE
    if t.startswith(("ITEM", "ARTIFACT", "ARTEFACT", "OBJECT", "PRODUCT", "WEAPON", "BOOK", "VẬT")):
        return ITEM
    if t.startswith(("ORG", "COMPANY", "FACTION", "GROUP", "EVENT", "DATE", "TIME", "CONCEPT", "SKILL", "POWER")):
        return CONCEPT
    if t.startswith(("NUMBER", "QUANTITY", "MONEY", "CARDINAL")):
        return NUMBER
    return PROPER


_INSTRUCTION = (
    "You extract entities and relations from segments of a novel (often "
    "Vietnamese). For EVERY numbered segment, list:\n"
    "- entities: characters, places, organizations, artifacts/items, powers, "
    "and significant dates or numbers. Use type labels like PERSON, LOCATION, "
    "ORGANIZATION, ITEM, EVENT, CONCEPT, DATE, NUMBER.\n"
    "- relations: subject-relation-object triples stated in that segment, with "
    "a SHORT_UPPERCASE_VERB label (e.g. OWNS, MENTORS, LOCATED_IN, DIED_IN, "
    "DISCOVERED, FRIENDS_WITH).\n"
    "Keep entity names exactly as written in the text (original language and "
    "diacritics). Extract only what is stated; do not infer or invent. Every "
    "input segment index must appear exactly once in the output."
)

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["name", "type"],
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "relation": {"type": "string"},
                                "object": {"type": "string"},
                            },
                            "required": ["subject", "relation", "object"],
                        },
                    },
                },
                "required": ["index", "entities", "relations"],
            },
        }
    },
    "required": ["segments"],
}


def _legacy_schema(node: Any) -> Any:
    """2.5-era responseSchema uses an uppercase Type enum."""
    if isinstance(node, list):
        return [_legacy_schema(n) for n in node]
    if isinstance(node, dict):
        return dict(
            (k, v.upper() if k == "type" and isinstance(v, str) else _legacy_schema(v))
            for k, v in node.items()
        )
    return node


def _thinking(model: str) -> Dict[str, Any]:
    if re.match(r"^gemini-2\.5-flash-lite", model):
        return {}  # thinking is off by default on lite
    if re.match(r"^gemini-2\.5", model):
        return {"thinkingConfig": {"thinkingBudget": 0}}
    if re.match(r"^gemini-3", model):
        return {"thinking_level": "minimal"}
    return {}


def _config_variants(model: str) -> List[Dict[str, Any]]:
    legacy = {
        "temperature": 0,
        "responseMimeType": "application/json",
        "responseSchema": _legacy_schema(_JSON_SCHEMA),
    }
    modern = {
        "temperature": 0,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": _JSON_SCHEMA},
    }
    bare = {"temperature": 0, "responseMimeType": "application/json"}
    for cfg in (legacy, modern, bare):
        cfg.update(_thinking(model))
    if re.match(r"^gemini-3", model):
        return [modern, legacy, bare]
    return [legacy, modern, bare]


class GeminiExtractorError(RuntimeError):
    pass


class GeminiExtractor(object):
    """Batch entity/relation extractor over the Gemini API."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EXTRACT_MODEL,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=API_TIMEOUT_S)
        self._variants = _config_variants(model)
        self._variant_index = 0

    def close(self) -> None:
        self._client.close()

    # -- public -------------------------------------------------------------

    def extract_segments(self, texts: Sequence[str]) -> List[SegmentExtraction]:
        """
        Extract entities + relations for each text. Chunks the workload to stay
        within request limits; raises GeminiExtractorError on failure (caller
        falls back to the local pipeline).
        """
        out: List[SegmentExtraction] = [SegmentExtraction([], []) for _ in texts]
        batch: List[int] = []
        batch_chars = 0
        for i, text in enumerate(texts):
            if batch and (len(batch) >= MAX_SEGMENTS_PER_CALL or batch_chars + len(text) > MAX_CHARS_PER_CALL):
                self._run_batch(texts, batch, out)
                batch, batch_chars = [], 0
            batch.append(i)
            batch_chars += len(text)
        if batch:
            self._run_batch(texts, batch, out)
        return out

    # -- internals ----------------------------------------------------------

    def _run_batch(self, texts: Sequence[str], indices: List[int], out: List[SegmentExtraction]) -> None:
        numbered = "\n\n".join("[%d]\n%s" % (i, texts[i]) for i in indices)
        raw = self._generate(numbered)
        parsed = self._parse(raw)
        for i in indices:
            if i in parsed:
                out[i] = parsed[i]

    def _generate(self, user_text: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(len(self._variants)):
            idx = (self._variant_index + attempt) % len(self._variants)
            body = {
                "systemInstruction": {"parts": [{"text": _INSTRUCTION}]},
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": self._variants[idx],
            }
            try:
                resp = self._client.post(
                    "%s/models/%s:generateContent" % (GEMINI_BASE, self.model),
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json=body,
                )
                if resp.status_code == 400:
                    # request-shape rejection: try the next config variant
                    last_error = GeminiExtractorError("Gemini API 400: %s" % resp.text[:200])
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text")
                )
                if not isinstance(text, str):
                    raise GeminiExtractorError("Gemini returned no JSON text")
                self._variant_index = idx
                return text
            except httpx.HTTPError as exc:
                # auth/quota/network: another shape will not fix it
                raise GeminiExtractorError("Gemini API error: %s" % exc)
        raise last_error or GeminiExtractorError("Gemini API: all request shapes rejected")

    @staticmethod
    def _parse(raw: str) -> Dict[int, SegmentExtraction]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except ValueError as exc:
            raise GeminiExtractorError("Gemini returned invalid JSON: %s" % exc)

        result: Dict[int, SegmentExtraction] = {}
        for seg in data.get("segments", []):
            try:
                index = int(seg.get("index"))
            except (TypeError, ValueError):
                continue
            entities: Dict[str, Mention] = {}
            for e in seg.get("entities", []):
                name = (e.get("name") or "").strip() if isinstance(e, dict) else ""
                if not name or len(name) > 80:
                    continue
                key = _canonical(name)
                if not key:
                    continue
                prev = entities.get(key)
                if prev is not None:
                    entities[key] = Mention(prev.key, prev.name, prev.type, prev.count + 1)
                else:
                    entities[key] = Mention(key, name, canonicalize_type(str(e.get("type", ""))), 1)
            relations: List[Dict[str, str]] = []
            for r in seg.get("relations", []):
                if not isinstance(r, dict):
                    continue
                subject = (r.get("subject") or "").strip()
                relation = (r.get("relation") or "").strip()
                obj = (r.get("object") or "").strip()
                if subject and relation and obj:
                    relations.append({"subject": subject, "relation": relation, "object": obj})
            result[index] = SegmentExtraction(list(entities.values()), relations)
        return result


def create_extractor(
    api_key: str,
    model: str = DEFAULT_EXTRACT_MODEL,
    logger=None,
) -> Optional[GeminiExtractor]:
    """Build the Gemini extractor if a key is configured; None otherwise."""
    if not api_key:
        return None
    try:
        return GeminiExtractor(api_key, model)
    except Exception as exc:  # pragma: no cover - construction is trivial
        if logger:
            logger("zero-mem: Gemini extractor unavailable (%s); using local NER." % exc)
        return None
