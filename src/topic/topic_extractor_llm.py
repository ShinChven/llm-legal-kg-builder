import concurrent.futures
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from google import genai

from src.topic.categories import (
    categories_as_prompt_text,
    is_valid_committee_topic,
    canonicalize_text,
    get_original_committee_name,
    get_original_topic_label,
)
from src.topic.act_topic_handler import ActTopicHandler


load_dotenv()


class Chunk(dict):
    pass


class TopicExtractor:
    """Core processor that extracts topics for a single Act by title using an LLM."""

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int = 1200,
        max_workers: int = 4,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        self.model_name = self._resolve_model_name(model_name)
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(
                f"Warning: No API key found in env var {api_key_env}. "
                "genai.Client will rely on default environment configuration."
            )
            self.genai_client = genai.Client()
        else:
            self.genai_client = genai.Client(api_key=api_key)
        self.handler = ActTopicHandler()
        print(f"[TopicExtractor] Using model: {self.model_name}")

    def _resolve_model_name(self, override: str | None) -> str:
        if override and override.strip():
            return override.strip()
        env_model = os.getenv("MODEL_NAME")
        if env_model and env_model.strip():
            return env_model.strip()
        raise ValueError(
            "No model specified. Set MODEL_NAME (e.g., 'gemini-2.5-pro') or pass model_name explicitly."
        )

    # ----- Public API -----
    def run_for_act(self, act_title: str) -> Dict:
        """Extract topics for the given act title, validate and store into DB. Returns result dict."""
        print(f"[TopicExtractor] Starting extraction for: {act_title}")
        text = self.handler.fetch_text_by_title(act_title)
        if not text:
            raise ValueError(f"Act '{act_title}' not found or has no text.")

        chunks = self._chunk_text(text)
        print(f"[TopicExtractor] Word count ~{len(text.split())}, chunks={len(chunks)}")

        llm_fn = self._create_llm_function(act_title)
        collected: Dict[Tuple[str, str], int] = self._process_chunks(
            chunks, llm_fn, act_title
        )

        # Soft synonym backstop: map common paraphrases to canonical topic labels
        synonyms = self._synonym_map()

        # Validate against predefined categories
        valid_items: List[Tuple[str, str, int]] = []
        invalid_items: List[Tuple[str, str, int]] = []
        for (committee, topic), score in collected.items():
            topic_key = canonicalize_text(topic)
            topic = synonyms.get(topic_key, topic)
            if is_valid_committee_topic(committee, topic):
                # map back to official display names for storage/printing
                oc = get_original_committee_name(committee) or committee
                ot = get_original_topic_label(committee, topic) or topic
                valid_items.append((oc, ot, max(0, min(100, int(score)))))
            else:
                invalid_items.append((committee, topic, score))

        if invalid_items:
            print(
                "[TopicExtractor] Dropping invalid committee-topic pairs (not in official list): "
                + "; ".join([f"{c} -> {t}" for (c, t, _s) in invalid_items])
            )

        # Store to DB
        upserted = self.handler.upsert_topics(act_title, valid_items)

        result = {
            "act_title": act_title,
            "topics": [
                {"committee": c, "topic": t, "importance": s} for (c, t, s) in valid_items
            ],
            "dropped": [
                {"committee": c, "topic": t, "importance": s} for (c, t, s) in invalid_items
            ],
            "upserted_count": upserted,
        }
        print(f"[TopicExtractor] Completed: upserted={upserted}, valid={len(valid_items)}, dropped={len(invalid_items)}")
        print(json.dumps(result, indent=2))
        return result

    # ----- Internals -----
    def _chunk_text(self, text: str) -> List[Chunk]:
        # Similar to relationship extractor: prefer line-aware, fallback to sentence-aware
        if self.chunk_size > 1_000_000:
            print(f"[TopicExtractor] Warning: Large chunk size {self.chunk_size}")

        if text.count("\n") <= 2:
            # sentence-aware
            sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\"\(\[])')
            raw_sentences = [s.strip() for s in sentence_pattern.split(text.strip()) if s.strip()]
            chunks: List[Chunk] = []
            buf: List[str] = []
            wc = 0

            def flush():
                nonlocal buf
                if buf:
                    chunks.append(Chunk({"text": " ".join(buf)}))
                    buf = []

            for s in raw_sentences:
                sw = len(s.split())
                if sw > self.chunk_size:
                    flush()
                    words = s.split()
                    for i in range(0, len(words), self.chunk_size):
                        chunks.append(Chunk({"text": " ".join(words[i : i + self.chunk_size])}))
                    wc = 0
                    continue
                if wc + sw > self.chunk_size and buf:
                    flush()
                    wc = 0
                buf.append(s)
                wc += sw
            flush()
            return chunks

        # line-aware
        lines = text.split("\n")
        chunks: List[Chunk] = []
        acc: List[str] = []
        wc = 0
        for ln in lines:
            lw = len(ln.split())
            if wc + lw > self.chunk_size and acc:
                chunks.append(Chunk({"text": "\n".join(acc)}))
                acc = []
                wc = 0
            acc.append(ln)
            wc += lw
        if acc:
            chunks.append(Chunk({"text": "\n".join(acc)}))
        return chunks

    def _process_chunks(
        self,
        chunks: List[Chunk],
        llm_function: Callable[[str, int], Dict],
        act_title: str,
    ) -> Dict[Tuple[str, str], int]:
        """Run LLM over chunks (multithreaded), merge by max importance per (committee, topic)."""
        aggregated: Dict[Tuple[str, str], int] = {}

        def task(chunk: Chunk, idx: int, total: int):
            print(f"[TopicExtractor] LLM for {act_title} chunk {idx+1}/{total}...")
            try:
                return llm_function(chunk["text"], total)
            except Exception as e:
                print(f"  chunk {idx+1}/{total} failed: {e}")
                return {"topics": []}

        total = len(chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = [exe.submit(task, ch, i, total) for i, ch in enumerate(chunks)]
            for fut in concurrent.futures.as_completed(futures):
                data = fut.result() or {}
                items = data.get("topics") or []
                for item in items:
                    c = item.get("committee", "").strip()
                    t_raw = item.get("topic", "").strip()
                    s = int(item.get("importance", 0))

                    # Split topics if they are comma-separated
                    topics = [topic.strip() for topic in t_raw.split(',')]

                    for t in topics:
                        if not t:
                            continue
                        key = (canonicalize_text(c), canonicalize_text(t))
                        # Keep the maximum score seen across chunks
                        aggregated[key] = max(aggregated.get(key, 0), max(0, min(100, s)))

        return aggregated

    def _create_llm_function(self, legislation_title: str) -> Callable[[str, int], Dict]:
        categories_text = categories_as_prompt_text()

        def llm_fn(text_chunk: str, total_chunks: int) -> Dict:
            if total_chunks > 1:
                passage_info = f"a chunk from the act '{legislation_title}'"
            else:
                passage_info = f"the full text of the act '{legislation_title}'"

            prompt = f"""You are an expert legal analyst for New Zealand legislation. Your task is Topic Classification.
Text to analyze ({passage_info}):
{text_chunk}



Classify the content into zero or more topics strictly from the official Select Committee categories below.
Only use the exact topics (responsibilities) listed under each committee. Do NOT invent new topics.

Official committees and their topics:
{categories_text}

Instructions:
- Read the text at the top carefully.
- Identify as many relevant topics as possible across any committees.
- Map paraphrases, synonyms, abbreviations, and variant spellings to the closest allowed topic label; then output the canonical label exactly as listed.
  Examples:
    • "IT", "ICT", "digital tech" -> "information technology"
    • "defense" -> "defence"
    • "consumer rights" -> "consumer protection and trading standards"
    • "emergency response" -> "civil defence and emergency management"
    • "road safety" -> "transport safety"
    • "arts and culture" -> two entries: "arts" and "culture and heritage"
- For each selected topic, assign an integer importance score on a 0–100 scale reflecting the topic’s relevance to this Act (higher = more central).
- Only choose topics that exactly match the provided list. If none match, return an empty list.
- Output JSON ONLY in the response_mime_type. No commentary.

Required JSON schema:
{{
  "act_title": "{legislation_title}",
  "topics": [
    {{"committee": "<committee name>", "topic": "<one allowed topic>", "importance": <0-100 integer>}},
    ...
  ]
}}
"""

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            try:
                raw = (response.text or "").strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                data = json.loads(raw)
                # normalize minimal shape for merging
                if not isinstance(data, dict):
                    return {"topics": []}
                if "topics" not in data or not isinstance(data["topics"], list):
                    return {"topics": []}
                return data
            except Exception as e:
                print(f"[TopicExtractor] JSON parse error: {e}")
                if hasattr(response, "text"):
                    print(f"  Raw response: {getattr(response, 'text', '')}")
                raise

        return llm_fn

    def _synonym_map(self) -> Dict[str, str]:
        """Return a small synonym-to-canonical-topic mapping (keys must be canonicalized)."""
        # Keys are canonicalized with canonicalize_text; values are canonical official topic labels.
        return {
            # Tech
            "it": "information technology",
            "ict": "information technology",
            "digital tech": "information technology",
            "digital technology": "information technology",
            # Defence / emergency
            "defense": "defence",
            "emergency response": "civil defence and emergency management",
            "emergency services": "civil defence and emergency management",
            # Consumer / finance
            "consumer rights": "consumer protection and trading standards",
            "banking finance": "banking and finance",
            "public finance": "government expenditure and financial performance",
            "public finances": "government expenditure and financial performance",
            "government spending": "government expenditure and financial performance",
            # Transport / infrastructure
            "transportation": "transport",
            "road safety": "transport safety",
            # Building
            "building construction": "building and construction",
            "construction": "building and construction",
            # Culture
            "arts culture": "arts",  # note: model is instructed to output two entries; this is a mild backstop
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract topics for a single Act title.")
    parser.add_argument("title", help="Exact Act title as stored in legislations.title")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--chunk", type=int, default=int(os.getenv("TOPIC_CHUNK_SIZE", "1200")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("TOPIC_MAX_WORKERS", "4")))
    args = parser.parse_args()

    extractor = TopicExtractor(model_name=args.model, chunk_size=args.chunk, max_workers=args.workers)
    extractor.run_for_act(args.title)
