from typing import Dict, List, Set, Tuple, Optional
import unicodedata


# Official NZ Parliament select committee categories and responsibilities.
# This structure is used both to validate LLM output and to compile into prompts so both
# prompt and program share the same categories.

# Map: Committee -> list of allowed topics (responsibilities)
COMMITTEE_CATEGORIES: Dict[str, List[str]] = {
    "Economic Development, Science and Innovation Committee": [
        "business development",
        "tourism",
        "crown minerals",
        "commerce",
        "consumer protection",
        "trading standards",
        "research",
        "science",
        "innovation",
        "intellectual property",
        "broadcasting",
        "communications",
        "information technology",
    ],
    "Education and Workforce Committee": [
        "education",
        "training",
        "employment",
        "immigration",
        "industrial relations",
        "health and safety",
        "accident compensation",
    ],
    "Environment Committee": [
        "conservation",
        "environment",
        "climate change",
    ],
    "Finance and Expenditure Committee": [
        "economic policy",
        "fiscal policy",
        "taxation",
        "revenue",
        "banking and finance",
        "superannuation",
        "insurance",
        "government expenditure",
        "financial performance",
        "public audit",
    ],
    "Foreign Affairs, Defence and Trade Committee": [
        "customs",
        "defence",
        "disarmament",
        "arms control",
        "foreign affairs",
        "trade",
        "veterans’ affairs",
    ],
    "Governance and Administration Committee": [
        "parliamentary services",
        "legislative services",
        "prime minister and cabinet",
        "state services",
        "statistics",
        "internal affairs",
        "civil defence",
        "emergency management",
        "local government",
    ],
    "Health Committee": [
        "health",
    ],
    "Justice Committee": [
        "constitutional matters",
        "electoral matters",
        "human rights",
        "justice",
        "courts",
        "crime",
        "criminal law",
        "police",
        "corrections",
        "crown legal services",
    ],
    "Māori Affairs Committee": [
        "māori affairs",
        "treaty of waitangi negotiations",
    ],
    "Primary Production Committee": [
        "agriculture",
        "biosecurity",
        "racing",
        "fisheries",
        "productive forestry",
        "lands",
        "land information",
    ],
    "Social Services and Community Committee": [
        "social development",
        "social housing",
        "income support",
        "women",
        "children",
        "young people",
        "seniors",
        "pacific peoples",
        "ethnic communities",
        "arts",
        "culture and heritage",
        "sport and recreation",
        "voluntary sector",
    ],
    "Transport and Infrastructure Committee": [
        "transport",
        "transport safety",
        "infrastructure",
        "energy",
        "building and construction",
    ],
}


def canonicalize_text(value: str) -> str:
    """Canonical form for case-insensitive matching, ASCII fold, punctuation softening, and whitespace normalization."""
    v = (value or "").strip().lower()
    # Replace common punctuation variants
    v = v.replace("’", "'")
    v = v.replace("&", " and ")
    v = v.replace("/", " and ")
    v = v.replace("-", " ")
    # Normalize accents/diacritics
    v_norm = unicodedata.normalize("NFKD", v)
    v_ascii = "".join(ch for ch in v_norm if not unicodedata.combining(ch))
    return " ".join(v_ascii.split())


def compile_allowed_pairs() -> Set[Tuple[str, str]]:
    """Return a set of (committee, topic) tuples in canonical form for fast validation."""
    pairs: Set[Tuple[str, str]] = set()
    for committee, topics in COMMITTEE_CATEGORIES.items():
        c = canonicalize_text(committee)
        for t in topics:
            pairs.add((c, canonicalize_text(t)))
    return pairs


def categories_as_prompt_text() -> str:
    """Render categories to a human-readable string for prompts (exact labels)."""
    lines: List[str] = []
    for committee, topics in COMMITTEE_CATEGORIES.items():
        lines.append(f"- {committee}: ")
        # Show topics as a comma-separated list in the exact canonical form to prevent drift
        lines.append("  " + ", ".join(topics))
    return "\n".join(lines)


def is_valid_committee_topic(committee: str, topic: str) -> bool:
    """Check if (committee, topic) pair is in the official list (case-insensitive)."""
    c = canonicalize_text(committee)
    t = canonicalize_text(topic)
    return (c, t) in compile_allowed_pairs()


def get_all_committees() -> List[str]:
    return list(COMMITTEE_CATEGORIES.keys())


def get_topics_for_committee(committee: str) -> List[str]:
    return COMMITTEE_CATEGORIES.get(committee, [])


def _committee_canon_map() -> Dict[str, str]:
    return {canonicalize_text(k): k for k in COMMITTEE_CATEGORIES.keys()}


def get_original_committee_name(committee_any_case: str) -> Optional[str]:
    """Return the official committee display name for the given committee label (case/diacritics-insensitive)."""
    return _committee_canon_map().get(canonicalize_text(committee_any_case))


def get_original_topic_label(committee_any_case: str, topic_any_case: str) -> Optional[str]:
    """Return the official topic label under the given committee if it exists; else None."""
    orig_committee = get_original_committee_name(committee_any_case)
    if not orig_committee:
        return None
    search = canonicalize_text(topic_any_case)
    for t in COMMITTEE_CATEGORIES.get(orig_committee, []):
        if canonicalize_text(t) == search:
            return t
    return None
