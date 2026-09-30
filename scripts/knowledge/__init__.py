"""
Structured educational content used by scripts/generate_dataset.py.

Each topic is a dict with the same schema (see `REQUIRED_TOPIC_FIELDS`). The
generator turns every field into one or more instruction -> response pairs,
so adding a topic here automatically adds a full set of training examples.
"""

from .ai_topics import AI_TOPICS
from .ml_topics import ML_TOPICS
from .dl_topics import DL_TOPICS
from .genai_topics import GENAI_TOPICS
from .comparisons import COMPARISONS

TOPICS: list[dict] = AI_TOPICS + ML_TOPICS + DL_TOPICS + GENAI_TOPICS

REQUIRED_TOPIC_FIELDS = (
    "key", "name", "short", "domain", "question", "definition", "simple",
    "eli10", "analogy", "how_it_works", "example", "applications",
    "advantages", "limitations", "key_points", "viva", "interview", "mcqs",
)

__all__ = ["TOPICS", "COMPARISONS", "REQUIRED_TOPIC_FIELDS"]
