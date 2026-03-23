from __future__ import annotations

import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# ============================================================
# Configuration
# ============================================================

WRITING_DIR = Path.home() / "writing"
NOVEL_AI_SCRIPT_PATH = WRITING_DIR / "ai" / "novel_ai" / "novel_ai.py"
NOVEL_PROJECT_DIR = WRITING_DIR / "novel_project"
PROJECT_MEMORY_DIR = NOVEL_PROJECT_DIR / "memory"
PROJECT_ANALYSIS_DIR = NOVEL_PROJECT_DIR / "analysis"
CHAPTERS_DIR = NOVEL_PROJECT_DIR / "chapters"
MANUSCRIPT_DIR = NOVEL_PROJECT_DIR / "manuscript"
MANUSCRIPT_PATH = MANUSCRIPT_DIR / "novel.txt"
CLEAN_MANUSCRIPT_PATH = MANUSCRIPT_DIR / "novel_clean.txt"
CONTINUITY_REPORTS_DIR = PROJECT_ANALYSIS_DIR / "continuity_reports"
REBUILD_LOG_DIR = PROJECT_ANALYSIS_DIR / "rebuild_logs"
CANON_MEMORY_PATH = PROJECT_MEMORY_DIR / "canon_memory.txt"
SCENE_SUMMARIES_PATH = PROJECT_MEMORY_DIR / "scene_summaries.txt"
IDEAS_PATH = PROJECT_MEMORY_DIR / "ideas.txt"
WORLD_RULES_PATH = PROJECT_MEMORY_DIR / "world.txt"
CHAPTER_FILENAME_PATTERN = re.compile(r"chapter_(\d+)\.txt$")
SUGGESTION_PATTERN = re.compile(
    r"^\s*(\d+)\.\s*(.+?)\s*(?:→|->)\s*\[([^\]]+)\]\s*$",
    re.MULTILINE,
)
CHAPTER_HEADER_PATTERN = re.compile(r"^CHAPTER\s+(\d+)\s*$")
CATEGORY_HEADER_PATTERN = re.compile(r"^\[(.+)\]\s*$")
FACT_STATE_PATTERN = re.compile(r"^(.*?)(?:\s*\((ACTIVE|RESOLVED)\))?$")
ANSI_ESCAPE_PATTERN = re.compile(r"\033\[[0-9;]*[A-Za-z]")
BRACKETED_PASTE_PATTERN = re.compile(r"(?:\033\[|\^\[\[?)(?:200~|201~|E)|\[\[200~|\[\[201~")
DISALLOWED_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAIN_TEMPERATURE = 0.8
SCENE_TEMPERATURE = 0.1
CONTINUITY_TEMPERATURE = 0.0
PROOFREAD_TEMPERATURE = 0.1
IDEA_RESURFACE_TEMPERATURE = 0.3

MAX_PREVIOUS_CHAPTERS = 3
MAX_SCENE_SUMMARIES = 5
MAX_CONVERSATION_TURNS = 6
MAX_CANON_CHARACTERS = 12000

MAIN_SYSTEM_PROMPT = """You are a thoughtful AI novel-writing assistant.
Help the user think through story ideas, scenes, structure, tone, character, and prose.
Use the provided memory carefully and naturally.
Be creative, clear, and practical.
Do not invent persistent facts unless the user states them.
"""

SCREENPLAY_SOURCE_PATH = NOVEL_PROJECT_DIR / "sources" / "screenplay.pdf"
ALLOWED_MEMORY_CATEGORIES = (
    "Character",
    "Relationship",
    "World",
    "Location",
    "Object",
    "Timeline",
    "Injury",
    "Mission State — Active",
    "Mission State — Resolved",
    "Psychological State — Active",
    "Psychological State — Resolved",
    "Relationship State — Active",
    "Relationship State — Resolved",
    "Technology State — Active",
    "Technology State — Resolved",
    "Foreshadowing Setup",
    "Foreshadowing Payoff",
)


ACTIVE_TO_RESOLVED_CATEGORY = {
    "Mission State — Active": "Mission State — Resolved",
    "Psychological State — Active": "Psychological State — Resolved",
    "Relationship State — Active": "Relationship State — Resolved",
    "Technology State — Active": "Technology State — Resolved",
}


SCENE_SYSTEM_PROMPT = """You are a Canon Memory Extraction Engine for a long-form novel system.

Your job is to read the provided chapter text and extract story-critical continuity facts,
even when they are implied through descriptive narrative.

Canon memory must preserve narrative lifecycle canon, not just flat facts.
Focus on what changes, escalates, resolves, or sets up future payoff.
It must NOT read like encyclopaedic worldbuilding notes.

Detect and preserve, when strongly supported by the text:

- Persistent world facts
- Character traits, behavioural signals, and injuries
- Relationship changes
- Mission condition changes
- Technology states
- Psychological pressure states
- Narrative tension activations
- Narrative tension resolutions
- Foreshadowing setups
- Foreshadowing payoffs

Prioritize, in roughly this order:

1. Mission risk, survival pressure, or major condition changes
2. Narrative tensions being activated, escalated, or resolved
3. Character psychological pressure, behaviour shifts, or revealing traits
4. Relationship tension, trust shifts, alliance changes, or responsibility shifts
5. Injuries or physical condition changes
6. Technology states affecting survival, access, timing, or the plot
7. Foreshadowing setups or payoffs
8. Persistent world rules or facts affecting later continuity
9. Important tracked objects or location changes

Deprioritize:

- Decorative environment description
- General setting layout
- Atmospheric flavour without continuity consequences
- Generic background lore
- Technical explanation unless it changes risk, capability, or future continuity

If the chapter is calm, store psychological, relational, or mission-progress movement instead.
Prefer fewer high-impact facts over many low-importance facts.
Each fact must be short, concrete, actionable, and non-duplicative.
Prefer narrative-active facts over descriptive facts.
Avoid extracting facts already implied by a stronger fact.
Do not invent or infer beyond what is strongly supported.

Encode narrative states inside the fact text itself while keeping the output format unchanged.
For example:
- Shield deployment mechanism jammed creating mission risk -> [World]
- Dr Manfrid showing hesitation under pressure indicating psychological strain -> [Character]
- Crew tension increases due to mission failure risk -> [Relationship]
- Impending neutron star event creates unresolved mission threat -> [Timeline]

Return output in this exact structure:

Memory suggestions:

1. Canon fact text -> [Category]
2. Canon fact text -> [Category]
3. Canon fact text -> [Category]

Allowed categories:

Character
Timeline
World
Object
Relationship
Injury
Location

Rules:

- Extract only strong continuity-relevant facts.
- Use exactly one allowed category per fact.
- Encode mission states, technology states, psychological pressure, tension activation or resolution, and foreshadowing inside the fact text instead of creating new categories.
- Detect cause-effect changes.
- Detect risk escalation.
- Detect responsibility shifts.
- Detect emotional or behavioural signals.
- Detect mission progress markers.
- Prefer facts that show a change, activation, escalation, resolution, setup, or payoff.
- Use short, concrete statements, not explanations.
- Keep only the minimum set of facts needed to preserve continuity.
- Do not repeat identical or meaningfully redundant facts.
- Do not invent.
- If no strong canon facts exist, return:

Memory suggestions:

None
"""

SCENE_SUMMARY_SYSTEM_PROMPT = """You are a strict isolated Narrative Analysis Engine for a novel-writing project.

You must analyze ONLY the material provided in this single request.
You must ignore chat history and never rely on prior conversation state.
Do not provide general writing advice, critique, or rewriting.
Use the canon memory, previous chapter summaries, and screenplay text only as reference material for comparison.
Do not invent unsupported facts.

Return output in this exact structure and order:

MEMORY SUGGESTIONS

1. Fact text → [Category]
2. Fact text → [Category]

CHAPTER STRUCTURE NOTE

<one of the required chapter-ending decisions>

SCREENPLAY ALIGNMENT NOTE

<alignment note>

RESOLUTION SUGGESTIONS

1. Short reference label
2. Short reference label

Rules for MEMORY SUGGESTIONS:
- Use a numbered list.
- Use short, concrete canon facts from the scene only.
- Only extract facts that matter for long-term continuity.
- Detect new canon facts introduced in the scene.
- Detect when a new unresolved narrative pressure, problem, mystery, danger, mission condition, relationship condition, psychological condition, or technology condition begins, and mark it as an Active state.
- Detect when a previously introduced problem, tension, mystery, danger, mission condition, relationship condition, psychological condition, or technology condition is resolved, and mark it as a Resolved state.
- Detect emotional or psychological turning points.
- Detect technology or mission condition changes.
- Detect setup or payoff of foreshadowing elements.
- Prioritize active mission problems, psychological changes, relationship shifts, injuries, technology state, location changes, foreshadowing, survival rules, and tracked objects.
- Deprioritize decorative description, layout, atmosphere, and generic lore.
- If the chapter is calm, prefer psychological or relational movement.
- Prefer fewer high-impact facts over many weak ones.
- Avoid duplicating facts already implied by stronger facts.
- Do NOT invent facts not present in the scene.
- Every suggestion must use exactly one of these categories: Character, Relationship, World, Location, Object, Timeline, Injury, Mission State — Active, Mission State — Resolved, Psychological State — Active, Psychological State — Resolved, Relationship State — Active, Relationship State — Resolved, Technology State — Active, Technology State — Resolved, Foreshadowing Setup, Foreshadowing Payoff.
- If there are no strong canon facts, return exactly:
MEMORY SUGGESTIONS

None

Rules for CHAPTER STRUCTURE NOTE:
- Evaluate chapter break strength based on whether the scene creates a strong location shift, tension pivot, emotional resolution, reveal, reversal, decision point, cliffhanger, or meaningful pause in the narrative lifecycle.
- Return exactly one of these lines:
Strong natural chapter ending point.
Moderate possible chapter ending.
Likely NOT a chapter ending.

Rules for SCREENPLAY ALIGNMENT NOTE:
- If screenplay text is not provided, say exactly: No screenplay source available for comparison.
- If screenplay text is provided and the scene diverges, briefly explain the divergence in motivation or events.
- If screenplay text is provided and aligned, say exactly: Scene remains consistent with screenplay intent.

Rules for RESOLUTION SUGGESTIONS:
- Look at canon memory facts marked ACTIVE.
- If the scene clearly resolves or concludes a situation, suggest it.
- Use a numbered list.
- Use a short reference label, not the full fact text.
- If no resolutions apply, write exactly:
None
"""

CONTINUITY_SYSTEM_PROMPT = """You are a strict continuity editor for a novel project.

Your task is to compare canon memory, previous chapters, and the selected chapter.
Return ONLY factual continuity issues.

Allowed issue types:
- injuries disappearing or changing without explanation
- characters appearing in incorrect locations
- timeline contradictions
- broken world rules
- object continuity errors
- relationship inconsistencies
- knowledge inconsistencies

Rules:
- Do not give writing advice.
- Do not rewrite any text.
- Do not praise the writing.
- Do not mention style, pacing, tone, or quality.
- Do not speculate beyond the provided material.
- If there are no continuity issues, say exactly: "No factual continuity issues found."

Return output in this exact structure:

CONTINUITY REPORT

- Issue 1
- Issue 2
"""

PROOFREAD_SYSTEM_PROMPT = """You are a professional proofreader.

You must:
- Correct grammar
- Correct punctuation
- Correct spelling
- Improve vocabulary where appropriate
- Improve sentence clarity

You must NOT:
- Change story meaning
- Add new content
- Remove content
- Summarise
- Change plot or character actions
- Reorder paragraphs
- Give writing advice
- Mention style or pacing

Return output in this exact structure:

PROOFREAD RESULT

<corrected text>

VOCABULARY SUGGESTIONS
- original → improved
"""



IDEA_RESURFACE_SYSTEM_PROMPT = """You are a strategic story editor.

Your job is to:
- Read the current chapter
- Read full canon memory
- Read the list of stored ideas

Then:

Select ONLY ideas that strongly fit the story at this point.

STRICT RULES:
- Do NOT force ideas
- Do NOT suggest weak connections
- Do NOT suggest ideas that don't naturally fit
- If confidence is low, return:

No ideas that could work here.

If ideas DO fit:

Return:

IDEA RESURFACING

1. <idea text> → <short reason why it fits now>
2. <idea text> → <short reason why it fits now>

Be concise.
No fluff.
No writing advice.
No rewriting."""


# ============================================================
# Filesystem helpers
# ============================================================


def ensure_project_files() -> None:
    """Create the expected project folders and files if they do not already exist."""
    PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (NOVEL_PROJECT_DIR / "sources").mkdir(parents=True, exist_ok=True)
    CONTINUITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REBUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CANON_MEMORY_PATH.touch(exist_ok=True)
    SCENE_SUMMARIES_PATH.touch(exist_ok=True)
    IDEAS_PATH.touch(exist_ok=True)
    WORLD_RULES_PATH.touch(exist_ok=True)



def atomic_write(path: Path, text: str) -> None:
    """Write text to a temporary file and atomically replace the destination."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)



def read_text_file(path: Path) -> str:
    """Read a text file and return cleaned text."""
    text = path.read_text(encoding="utf-8").strip()
    return text if text else "(empty)"



def load_pdf_text(path: Path) -> str:
    """Extract text from a PDF if possible, otherwise return a fallback note."""
    if not path.exists() or not path.is_file():
        return ""

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return "(screenplay PDF exists but no PDF reader library is installed)"

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            cleaned_page_text = page_text.strip()
            if cleaned_page_text:
                pages.append(cleaned_page_text)
        return "\n\n".join(pages).strip()
    except Exception as exc:
        return f"(could not read screenplay PDF: {exc})"


def load_memory_block() -> str:
    """Load canonical story memory for the main assistant and continuity checker."""
    canon_text = read_text_file(CANON_MEMORY_PATH)
    if len(canon_text) > MAX_CANON_CHARACTERS:
        return canon_text[-MAX_CANON_CHARACTERS:]
    return canon_text


def load_world_rules_block() -> str:
    """Load structured world rules for continuity checking."""
    if not WORLD_RULES_PATH.exists():
        return "(empty)"
    return read_text_file(WORLD_RULES_PATH)


def load_ideas_block() -> str:
    """Load saved writing ideas for idea suggestion requests."""
    if not IDEAS_PATH.exists():
        return ""
    return IDEAS_PATH.read_text(encoding="utf-8").strip()


def load_previous_scene_summaries_block() -> str:
    """Load previous scene summaries for isolated narrative analysis context."""
    if not SCENE_SUMMARIES_PATH.exists():
        return ""

    divider = "=" * 40
    summaries_text = SCENE_SUMMARIES_PATH.read_text(encoding="utf-8").strip()
    if not summaries_text:
        return ""

    parts = summaries_text.split(divider)
    entries: list[str] = []
    index = 1
    while index + 1 < len(parts):
        header = parts[index].strip()
        body = parts[index + 1].strip()
        if header and body:
            entries.append(f"{divider}\n{header}\n{divider}\n{body}")
        index += 2

    if not entries:
        return summaries_text

    return "\n\n".join(entries[-MAX_SCENE_SUMMARIES:])


def load_screenplay_block() -> str:
    """Load screenplay source text from PDF when available."""
    return load_pdf_text(SCREENPLAY_SOURCE_PATH)



def append_world_rule(category: str, rule_text: str) -> None:
    """Append one structured world rule without overwriting existing rules."""
    PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_category = category.strip()
    cleaned_rule_text = rule_text.strip()

    if not cleaned_category or not cleaned_rule_text:
        raise ValueError("Category and rule text are required.")

    entry = f"[{cleaned_category}]\n{cleaned_rule_text}"
    existing_content = (
        WORLD_RULES_PATH.read_text(encoding="utf-8")
        if WORLD_RULES_PATH.exists()
        else ""
    )

    with WORLD_RULES_PATH.open("a", encoding="utf-8") as file:
        if existing_content.strip():
            separator = "\n\n"
            if existing_content.endswith("\n\n"):
                separator = ""
            elif existing_content.endswith("\n"):
                separator = "\n"
            file.write(separator)
        file.write(entry + "\n")



def parse_canon_fact(fact_line: str) -> dict[str, str]:
    """Parse one canon fact line into text and lifecycle state."""
    match = FACT_STATE_PATTERN.fullmatch(fact_line.strip())
    if match is None:
        return {"text": fact_line.strip(), "state": "ACTIVE"}

    fact_text = match.group(1).strip()
    fact_state = (match.group(2) or "ACTIVE").strip().upper()
    return {
        "text": fact_text,
        "state": "RESOLVED" if fact_state == "RESOLVED" else "ACTIVE",
    }



def render_canon_fact(fact: str | dict[str, Any]) -> str:
    """Render one canon fact entry with its lifecycle state marker."""
    if isinstance(fact, dict):
        fact_text = str(fact.get("text", "")).strip()
        fact_state = str(fact.get("state", "ACTIVE")).strip().upper() or "ACTIVE"
    else:
        fact_text = str(fact).strip()
        fact_state = "ACTIVE"

    normalized_state = "RESOLVED" if fact_state == "RESOLVED" else "ACTIVE"
    return f"{fact_text} ({normalized_state})"



def get_fact_text(fact: str | dict[str, Any]) -> str:
    """Return canon fact text without any lifecycle marker."""
    if isinstance(fact, dict):
        return str(fact.get("text", "")).strip()
    return parse_canon_fact(str(fact)).get("text", "").strip()



def parse_canon_memory(content: str) -> list[dict[str, Any]]:
    """Parse canon memory text into ordered chapter/category blocks."""
    chapters: list[dict[str, Any]] = []
    current_chapter: dict[str, Any] | None = None
    current_category: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped or stripped == "========================":
            continue

        chapter_match = CHAPTER_HEADER_PATTERN.fullmatch(stripped)
        if chapter_match is not None:
            current_chapter = {
                "number": int(chapter_match.group(1)),
                "categories": OrderedDict(),
            }
            chapters.append(current_chapter)
            current_category = None
            continue

        category_match = CATEGORY_HEADER_PATTERN.fullmatch(stripped)
        if category_match is not None:
            if current_chapter is None:
                continue
            current_category = category_match.group(1).strip()
            current_chapter["categories"].setdefault(current_category, [])
            continue

        if stripped.startswith("-") and current_chapter is not None and current_category is not None:
            fact = stripped[1:].strip()
            if fact:
                current_chapter["categories"][current_category].append(parse_canon_fact(fact))

    return chapters



def render_canon_memory(chapters: list[dict[str, Any]]) -> str:
    """Render ordered chapter/category blocks back to canon memory text."""
    blocks: list[str] = []
    for chapter in chapters:
        lines = ["========================", f"CHAPTER {chapter['number']}"]
        for category, facts in chapter["categories"].items():
            lines.append("")
            lines.append(f"[{category}]")
            for fact in facts:
                lines.append(f"- {render_canon_fact(fact)}")
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks).rstrip() + ("\n" if blocks else "")


def order_memory_categories(categories: OrderedDict[str, list[dict[str, str]]]) -> OrderedDict[str, list[dict[str, str]]]:
    """Return categories in canonical display order, keeping unknown categories last."""
    ordered_categories: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    for category in ALLOWED_MEMORY_CATEGORIES:
        if category in categories and categories[category]:
            ordered_categories[category] = categories[category]

    for category, facts in categories.items():
        if category not in ordered_categories and facts:
            ordered_categories[category] = facts

    return ordered_categories



def normalize_fact_text(text: str | dict[str, Any]) -> str:
    """Normalize canon fact text for duplicate and near-duplicate comparison."""
    cleaned_text = get_fact_text(text).lower().strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip("\"'")
    cleaned_text = cleaned_text.replace("--", "")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.rstrip(".,;: ")
    return cleaned_text


def facts_are_similar(fact_a: str | dict[str, Any], fact_b: str | dict[str, Any]) -> bool:
    """Return True when two canon facts are duplicates or near-duplicates."""
    normalized_fact_a = normalize_fact_text(fact_a)
    normalized_fact_b = normalize_fact_text(fact_b)

    if normalized_fact_a == normalized_fact_b:
        return True

    words_a = normalized_fact_a.split()
    words_b = normalized_fact_b.split()
    if not words_a or not words_b:
        return False

    common_words = len(set(words_a) & set(words_b))
    overlap_ratio = common_words / max(len(words_a), len(words_b))
    return overlap_ratio >= 0.6


def append_to_canon_memory(
    chapter_number: int,
    selected_facts: list[tuple[str, str]],
) -> None:
    """Append selected canon facts into the appropriate chapter block."""
    ensure_project_files()
    cleaned_facts = [
        (fact.strip(), category.strip())
        for fact, category in selected_facts
        if fact.strip() and category.strip()
    ]
    if not cleaned_facts:
        print("No canon facts selected to save.")
        return

    existing_content = CANON_MEMORY_PATH.read_text(encoding="utf-8")
    chapters = parse_canon_memory(existing_content)

    target_chapter: dict[str, Any] | None = None
    for chapter in chapters:
        if chapter["number"] == chapter_number:
            target_chapter = chapter
            break

    if target_chapter is None:
        target_chapter = {"number": chapter_number, "categories": OrderedDict()}
        chapters.append(target_chapter)

    saved_count = 0
    skipped_duplicates = 0
    for fact, category in cleaned_facts:
        category_facts = target_chapter["categories"].setdefault(category, [])
        if any(facts_are_similar(existing_fact, fact) for existing_fact in category_facts):
            skipped_duplicates += 1
            print(f"Skipped duplicate canon fact: {fact}")
            continue
        category_facts.append({"text": fact, "state": "ACTIVE"})
        saved_count += 1

    if saved_count > 0:
        target_chapter["categories"] = order_memory_categories(target_chapter["categories"])
        chapters.sort(key=lambda chapter: chapter["number"])
        atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))

    print(f"Saved {saved_count} canon fact(s).")
    print(f"Skipped {skipped_duplicates} duplicate fact(s).")



def mark_fact_resolved(chapter_number: int, category: str, fact_text: str) -> bool:
    """Mark the first matching canon fact in a chapter/category as resolved."""
    ensure_project_files()
    if not CANON_MEMORY_PATH.exists():
        return False

    cleaned_category = category.strip()
    cleaned_fact_text = fact_text.strip()
    if not cleaned_category or not cleaned_fact_text:
        return False

    chapters = parse_canon_memory(CANON_MEMORY_PATH.read_text(encoding="utf-8"))
    updated = False

    for chapter in chapters:
        if chapter["number"] != chapter_number:
            continue

        category_facts = chapter["categories"].get(cleaned_category, [])
        for existing_fact in category_facts:
            if not facts_are_similar(existing_fact, cleaned_fact_text):
                continue
            if existing_fact.get("state") == "RESOLVED":
                return False
            existing_fact["state"] = "RESOLVED"
            updated = True
            break
        break

    if not updated:
        return False

    atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))
    return True



def append_scene_summary(chapter_number: int, summary_text: str) -> None:
    """Append full narrative analysis output to the scene summaries log."""
    ensure_project_files()
    cleaned_summary = summary_text.strip()
    if not cleaned_summary:
        return

    divider = "=" * 40
    entry = f"{divider}\nCHAPTER {chapter_number}\n{divider}\n{cleaned_summary}\n"

    with SCENE_SUMMARIES_PATH.open("a", encoding="utf-8") as file:
        if SCENE_SUMMARIES_PATH.stat().st_size > 0:
            file.write("\n")
        file.write(entry)


def remove_scene_summary_block(chapter_number: int) -> bool:
    """Remove one chapter block from the scene summaries log if it exists."""
    if not SCENE_SUMMARIES_PATH.exists():
        return False

    content = SCENE_SUMMARIES_PATH.read_text(encoding="utf-8")
    if not content.strip():
        return False

    divider = "=" * 40
    escaped_divider = re.escape(divider)
    block_pattern = re.compile(
        rf"(?:^|\n){escaped_divider}\nCHAPTER {chapter_number}\n{escaped_divider}\n.*?"
        rf"(?=\n{escaped_divider}\nCHAPTER \d+\n{escaped_divider}\n|\Z)",
        re.DOTALL,
    )
    updated_content, removals = block_pattern.subn("", content, count=1)
    if removals == 0:
        return False

    normalized_content = updated_content.strip()
    atomic_write(
        SCENE_SUMMARIES_PATH,
        (normalized_content + "\n") if normalized_content else "",
    )
    return True



def append_idea(text: str) -> None:
    """Append a timestamped writing idea to the ideas log."""
    PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()
    with IDEAS_PATH.open("a", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "========================",
                    timestamp.strftime("%Y-%m-%d %H:%M"),
                    "========================",
                    "Idea:",
                    text.strip(),
                    "",
                    "",
                ]
            )
        )



def build_chapter_memory_block(
    chapter_number: int,
    suggestions: list[tuple[int, str, str]],
) -> dict[str, Any]:
    """Convert extracted suggestions into one ordered canon-memory chapter block."""
    categories: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for _, fact, category in suggestions:
        cleaned_fact = fact.strip()
        cleaned_category = category.strip()
        if not cleaned_fact or not cleaned_category:
            continue
        category_facts = categories.setdefault(cleaned_category, [])
        if any(facts_are_similar(existing_fact, cleaned_fact) for existing_fact in category_facts):
            continue
        category_facts.append({"text": cleaned_fact, "state": "ACTIVE"})

    return {
        "number": chapter_number,
        "categories": order_memory_categories(categories),
    }



def insert_or_replace_chapter_block(
    chapters: list[dict[str, Any]],
    rebuilt_chapter: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace one chapter block and keep chapter ordering numeric."""
    filtered_chapters = [
        chapter
        for chapter in chapters
        if chapter["number"] != rebuilt_chapter["number"]
    ]
    filtered_chapters.append(rebuilt_chapter)
    filtered_chapters.sort(key=lambda chapter: chapter["number"])
    return filtered_chapters



def write_rebuild_log(
    mode: str,
    lines: list[str],
) -> Path:
    """Write a rebuild log file and return its path."""
    timestamp = datetime.utcnow()
    log_path = REBUILD_LOG_DIR / f"rebuild_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    content = "\n".join(
        [
            f"mode: {mode}",
            *lines,
            f"timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
    )
    atomic_write(log_path, content + "\n")
    return log_path



def extract_chapter_number(path: Path) -> int | None:
    """Return the chapter number from a chapter filename."""
    match = CHAPTER_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))



def load_sorted_chapter_paths() -> list[Path]:
    """Return project chapter files sorted numerically."""
    chapter_paths = []
    for path in CHAPTERS_DIR.iterdir():
        if not path.is_file():
            continue
        chapter_number = extract_chapter_number(path)
        if chapter_number is None:
            continue
        chapter_paths.append((chapter_number, path))
    chapter_paths.sort(key=lambda item: item[0])
    return [path for _, path in chapter_paths]



def format_chapter_block(chapter_paths: list[Path]) -> str:
    """Read chapter files into one labeled text block."""
    sections = []
    for path in chapter_paths:
        sections.append(f"{path.name}:\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(sections) if sections else "(none)"



def build_manuscript_text(chapter_paths: list[Path]) -> str:
    """Compile numbered chapter files into one manuscript string."""
    sections: list[str] = []

    for path in chapter_paths:
        chapter_number = extract_chapter_number(path)
        if chapter_number is None:
            continue

        chapter_text = path.read_text(encoding="utf-8").strip()
        sections.append(
            "\n".join(
                [
                    "========================",
                    f"CHAPTER {chapter_number}",
                    "========================",
                    "",
                    chapter_text,
                ]
            ).rstrip()
        )

    return "\n\n".join(sections) + ("\n\n" if sections else "")


def clean_manuscript_text(text: str) -> str:
    """Normalize chapter text for clean manuscript output."""
    cleaned_text = ANSI_ESCAPE_PATTERN.sub("", text)
    cleaned_text = BRACKETED_PASTE_PATTERN.sub("", cleaned_text)
    cleaned_text = DISALLOWED_CONTROL_CHAR_PATTERN.sub("", cleaned_text)

    cleaned_lines: list[str] = []
    previous_blank = False

    for raw_line in cleaned_text.splitlines():
        stripped_line = raw_line.rstrip()

        if re.fullmatch(r"[-=*]{3,}", stripped_line.strip()):
            continue

        if not stripped_line.strip():
            if previous_blank:
                continue
            cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(stripped_line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def build_clean_manuscript_text(chapter_paths: list[Path]) -> str:
    """Compile chapter files into a cleaned manuscript string."""
    sections: list[str] = []

    for path in chapter_paths:
        chapter_number = extract_chapter_number(path)
        if chapter_number is None:
            continue

        chapter_text = clean_manuscript_text(path.read_text(encoding="utf-8"))
        sections.append(f"CHAPTER {chapter_number}\n\n{chapter_text}".rstrip())

    return "\n\n\n".join(sections) + ("\n" if sections else "")


# ============================================================
# Input helpers
# ============================================================


def clean_terminal_text(text: str) -> str:
    """Remove terminal control artifacts from pasted multi-line text."""
    cleaned_text = ANSI_ESCAPE_PATTERN.sub("", text)
    cleaned_text = BRACKETED_PASTE_PATTERN.sub("", cleaned_text)
    cleaned_text = DISALLOWED_CONTROL_CHAR_PATTERN.sub("", cleaned_text)
    return cleaned_text.strip()


def collect_multiline_input(end_marker: str = "END") -> str:
    """Collect multi-line input until the user types the end marker."""
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            print()
            break

        if line.strip() == end_marker:
            break
        lines.append(line)

    return clean_terminal_text("\n".join(lines))



def prompt_for_chapter_number(prompt_text: str = "Chapter number?") -> int | None:
    """Ask the user for a chapter number."""
    print(prompt_text)
    try:
        raw_value = input("> ").strip()
    except EOFError:
        print()
        return None

    if not raw_value:
        print("No chapter number entered.")
        return None

    if not raw_value.isdigit():
        print("Chapter number must be a positive integer.")
        return None

    chapter_number = int(raw_value)
    if chapter_number <= 0:
        print("Chapter number must be a positive integer.")
        return None

    return chapter_number



def prompt_for_confirmation(prompt_text: str) -> bool:
    """Ask the user a yes/no question and return True for yes."""
    print(prompt_text)
    try:
        response = input("> ").strip().lower()
    except EOFError:
        print()
        return False

    return response == "y"



def parse_memory_suggestions(result: str) -> list[tuple[int, str, str]]:
    """Parse numbered memory suggestions from the scene extractor output."""
    suggestions: list[tuple[int, str, str]] = []
    for match in SUGGESTION_PATTERN.finditer(result):
        number = int(match.group(1))
        fact = match.group(2).strip()
        category = match.group(3).strip()
        if category not in ALLOWED_MEMORY_CATEGORIES:
            continue
        suggestions.append((number, fact, category))
    return suggestions



def parse_resolution_suggestions(result_text: str) -> list[tuple[int, str]]:
    """Parse numbered resolution suggestions from the scene summary output."""
    match = re.search(
        r"RESOLUTION SUGGESTIONS\s*(.*?)(?:\n[A-Z][A-Z ]+\n|\Z)",
        result_text,
        re.DOTALL,
    )
    if match is None:
        return []

    section_text = match.group(1).strip()
    if not section_text or section_text == "None":
        return []

    suggestions: list[tuple[int, str]] = []
    for line in section_text.splitlines():
        numbered_match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if numbered_match is None:
            continue
        suggestions.append((int(numbered_match.group(1)), numbered_match.group(2).strip()))
    return suggestions



def prompt_for_selection(max_number: int, prompt_text: str = "\nSelect numbers to save:") -> list[int] | None:
    """Ask the user which numbered suggestions should be applied."""
    print(prompt_text)
    try:
        raw_selection = input("> ").strip()
    except EOFError:
        print()
        return None

    if not raw_selection:
        print("No selections entered. Nothing saved.")
        return []

    tokens = raw_selection.replace(",", " ").split()
    selections: list[int] = []
    seen: set[int] = set()

    for token in tokens:
        if not token.isdigit():
            print(f"Invalid selection: {token}")
            return None
        number = int(token)
        if number < 1 or number > max_number:
            print(f"Selection out of range: {number}")
            return None
        if number not in seen:
            selections.append(number)
            seen.add(number)

    return selections


def resolution_label_matches_fact(label: str, fact: str) -> bool:
    """Return True when a short resolution label clearly points to an ACTIVE fact."""
    normalized_label = normalize_fact_text(label)
    normalized_fact = normalize_fact_text(fact)
    if not normalized_label or not normalized_fact:
        return False

    if normalized_label in normalized_fact or normalized_fact in normalized_label:
        return True

    label_words = normalized_label.split()
    if label_words and all(word in normalized_fact.split() for word in label_words):
        return True

    return facts_are_similar(label, fact)



def apply_resolutions(selected_labels: list[str]) -> int:
    """Move matching ACTIVE canon facts into their RESOLVED categories."""
    ensure_project_files()
    if not selected_labels or not CANON_MEMORY_PATH.exists():
        return 0

    canon_content = CANON_MEMORY_PATH.read_text(encoding="utf-8")
    chapters = parse_canon_memory(canon_content)
    resolved_count = 0

    for label in selected_labels:
        normalized_label = normalize_fact_text(label)
        if not normalized_label:
            continue

        matched_entry: tuple[dict[str, Any], str, str] | None = None

        for chapter in chapters:
            for category in ACTIVE_TO_RESOLVED_CATEGORY:
                facts = chapter["categories"].get(category, [])
                for fact in facts:
                    if resolution_label_matches_fact(label, fact):
                        matched_entry = (chapter, category, fact)
                        break
                if matched_entry is not None:
                    break
            if matched_entry is not None:
                break

        if matched_entry is None:
            continue

        chapter, active_category, fact = matched_entry
        resolved_category = ACTIVE_TO_RESOLVED_CATEGORY[active_category]
        active_facts = chapter["categories"].get(active_category, [])
        if fact not in active_facts:
            continue

        active_facts.remove(fact)
        resolved_facts = chapter["categories"].setdefault(resolved_category, [])
        if fact not in resolved_facts:
            resolved_facts.append(fact)
        if not active_facts:
            chapter["categories"].pop(active_category, None)

        chapter["categories"] = order_memory_categories(chapter["categories"])
        resolved_count += 1

    if resolved_count > 0:
        chapters.sort(key=lambda chapter: chapter["number"])
        atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))

    return resolved_count


# ============================================================
# OpenAI helpers
# ============================================================


def create_client() -> Any:
    """Create the OpenAI client using environment configuration."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI package is not installed. Run: pip install openai"
        ) from exc

    return OpenAI()



def request_chat_completion(
    client: Any,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    """Send a chat request and return plain text output."""
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        temperature=temperature,
    )
    return response.output_text.strip()


# ============================================================
# Prompt builders
# ============================================================


def build_main_messages(
    memory_block: str,
    conversation_history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    """Build the message list for normal assistant chat."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": MAIN_SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": f"Persistent novel memory:\n\n{memory_block}",
        },
    ]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    return messages



def build_scene_messages(
    scene_text: str,
) -> list[dict[str, str]]:
    """Build the isolated message list for scene extraction."""
    return [
        {
            "role": "system",
            "content": SCENE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": scene_text,
        },
    ]


def build_scene_summary_messages(
    scene_text: str,
    canon_memory_block: str,
    previous_summaries_block: str,
    screenplay_block: str,
) -> list[dict[str, str]]:
    """Build the isolated message list for narrative scene analysis."""
    screenplay_text = screenplay_block.strip() if screenplay_block.strip() else "(none)"
    previous_summaries_text = (
        previous_summaries_block.strip() if previous_summaries_block.strip() else "(none)"
    )

    return [
        {
            "role": "system",
            "content": SCENE_SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Scene text:\n\n{scene_text}\n\n"
                f"Canon memory:\n\n{canon_memory_block}\n\n"
                f"Previous chapter summaries:\n\n{previous_summaries_text}\n\n"
                f"Screenplay text:\n\n{screenplay_text}"
            ),
        },
    ]



def build_continuity_messages(
    memory_block: str,
    world_rules_block: str,
    previous_chapters_block: str,
    selected_chapter_name: str,
    selected_chapter_text: str,
) -> list[dict[str, str]]:
    """Build the isolated message list for continuity checking."""
    return [
        {
            "role": "system",
            "content": CONTINUITY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Canon memory:\n\n{memory_block}\n\n"
                f"World rules:\n\n{world_rules_block}\n\n"
                f"Previous chapters:\n\n{previous_chapters_block}\n\n"
                f"Selected chapter ({selected_chapter_name}):\n\n{selected_chapter_text}"
            ),
        },
    ]


def build_proofread_messages(text_to_proofread: str) -> list[dict[str, str]]:
    """Build the isolated message list for proofreading."""
    return [
        {
            "role": "system",
            "content": PROOFREAD_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text_to_proofread,
        },
    ]




def build_idea_resurface_messages(
    chapter_text: str,
    canon_memory_block: str,
    ideas_block: str,
) -> list[dict[str, str]]:
    """Build the message list for resurfacing highly relevant saved ideas."""
    return [
        {
            "role": "system",
            "content": IDEA_RESURFACE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Current chapter text:\n\n{chapter_text}\n\n"
                f"Canon memory:\n\n{canon_memory_block}\n\n"
                f"Stored ideas:\n\n{ideas_block}"
            ),
        },
    ]


# ============================================================
# Extraction helpers
# ============================================================


def extract_memory_suggestions_for_text(
    client: Any,
    scene_text: str,
) -> list[tuple[int, str, str]]:
    """Run the shared isolated extractor and return parsed suggestions."""
    result = request_chat_completion(
        client=client,
        messages=build_scene_messages(scene_text),
        temperature=SCENE_TEMPERATURE,
    )
    return parse_memory_suggestions(result)


def generate_scene_summary_for_chapter(
    client: Any,
    chapter_number: int,
) -> str:
    """Run the scene summary prompt for one chapter file."""
    chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
    if not chapter_path.exists() or not chapter_path.is_file():
        raise FileNotFoundError(f"Chapter file not found: {chapter_path}")

    scene_text = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
    if not scene_text:
        raise ValueError(f"Chapter {chapter_number} is empty.")

    return request_chat_completion(
        client=client,
        messages=build_scene_summary_messages(
            scene_text=scene_text,
            canon_memory_block=load_memory_block(),
            previous_summaries_block=load_previous_scene_summaries_block(),
            screenplay_block=load_screenplay_block(),
        ),
        temperature=SCENE_TEMPERATURE,
    )



# ============================================================
# Command handlers
# ============================================================


def handle_scene_summary(client: Any) -> None:
    """Analyse one pasted scene in a fully isolated request."""
    ensure_project_files()
    chapter_number = prompt_for_chapter_number()
    if chapter_number is None:
        return

    print("Paste scene. Type END on new line when finished.")
    scene_text = collect_multiline_input(end_marker="END")
    scene_text = clean_terminal_text(scene_text)

    if not scene_text:
        print("No scene entered.")
        return

    canon_memory_block = load_memory_block()
    previous_summaries_block = load_previous_scene_summaries_block()
    screenplay_block = load_screenplay_block()

    try:
        result = request_chat_completion(
            client=client,
            messages=build_scene_summary_messages(
                scene_text=scene_text,
                canon_memory_block=canon_memory_block,
                previous_summaries_block=previous_summaries_block,
                screenplay_block=screenplay_block,
            ),
            temperature=SCENE_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Scene summary failed: {exc}")
        return

    print()
    print(result)

    suggestions = parse_memory_suggestions(result)
    if not suggestions:
        print("\nNo structured canon suggestions found. Nothing saved.")
    else:
        selection_numbers = prompt_for_selection(
            max(number for number, _, _ in suggestions)
        )
        if selection_numbers is None:
            print("Nothing saved.")
        else:
            selected_lookup = {number for number in selection_numbers}
            selected_facts = [
                (fact, category)
                for number, fact, category in suggestions
                if number in selected_lookup
            ]

            if selected_facts:
                try:
                    append_to_canon_memory(chapter_number, selected_facts)
                except OSError as exc:
                    print(f"Could not save canon memory: {exc}")
                    return
            else:
                print("Nothing saved.")

    resolution_suggestions = parse_resolution_suggestions(result)
    if resolution_suggestions:
        print("\nSelect resolutions to apply:")
        for number, label_text in resolution_suggestions:
            print(f"{number}. {label_text}")

        resolution_numbers = prompt_for_selection(
            max(number for number, _ in resolution_suggestions),
            prompt_text="",
        )
        if resolution_numbers is None:
            print("Resolved 0 canon facts.")
        else:
            selected_resolution_numbers = {number for number in resolution_numbers}
            selected_labels = [
                label_text
                for number, label_text in resolution_suggestions
                if number in selected_resolution_numbers
            ]
            try:
                resolved_count = apply_resolutions(selected_labels)
            except OSError as exc:
                print(f"Could not update canon memory resolutions: {exc}")
                return
            print(f"Resolved {resolved_count} canon facts.")

    try:
        append_scene_summary(chapter_number, result)
    except OSError as exc:
        print(f"Narrative analysis generated, but could not save log: {exc}")
        return

    print(f"Narrative analysis log saved to {SCENE_SUMMARIES_PATH}.")



def handle_continuity_check(client: Any) -> None:
    """Run a continuity report in a fully isolated request."""
    ensure_project_files()
    print("Enter chapter filename:")
    try:
        selected_name = input("> ").strip()
    except EOFError:
        print()
        return

    if not selected_name:
        print("No chapter filename entered.")
        return

    if not CHAPTERS_DIR.exists():
        print(f"Missing chapters directory: {CHAPTERS_DIR}")
        return

    chapter_paths = load_sorted_chapter_paths()
    selected_path = CHAPTERS_DIR / selected_name

    if selected_path not in chapter_paths:
        print("Chapter not found.")
        return

    selected_number = extract_chapter_number(selected_path)
    if selected_number is None:
        print("Invalid chapter filename format.")
        return

    previous_paths = [
        path
        for path in chapter_paths
        if extract_chapter_number(path) is not None
        and extract_chapter_number(path) < selected_number
    ][-MAX_PREVIOUS_CHAPTERS:]

    memory_block = load_memory_block()
    world_rules_block = load_world_rules_block()
    previous_chapters_block = format_chapter_block(previous_paths)
    selected_chapter_text = selected_path.read_text(encoding="utf-8").strip()
    messages = build_continuity_messages(
        memory_block=memory_block,
        world_rules_block=world_rules_block,
        previous_chapters_block=previous_chapters_block,
        selected_chapter_name=selected_path.name,
        selected_chapter_text=selected_chapter_text,
    )

    try:
        report = request_chat_completion(
            client=client,
            messages=messages,
            temperature=CONTINUITY_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Continuity check failed: {exc}")
        return

    report_path = CONTINUITY_REPORTS_DIR / f"{selected_path.stem}_report.txt"
    try:
        atomic_write(report_path, report + "\n")
    except OSError as exc:
        print(f"Continuity report could not be saved: {exc}")
        return

    print()
    print(report)
    print()
    print(f"Continuity report saved to {report_path}.")



def handle_rebuild_memory(client: Any) -> None:
    """Rebuild canon memory for the whole novel or one chapter."""
    ensure_project_files()
    print("Rebuild options:")
    print()
    print("1) Rebuild entire novel")
    print("2) Rebuild single chapter")

    try:
        selection = input("> ").strip()
    except EOFError:
        print()
        return

    if selection == "1":
        if not CHAPTERS_DIR.exists():
            print(f"Missing chapters directory: {CHAPTERS_DIR}")
            return

        chapter_paths = load_sorted_chapter_paths()
        if not chapter_paths:
            print("No chapter files found.")
            return

        rebuilt_chapters: list[dict[str, Any]] = []
        total_facts = 0

        for chapter_path in chapter_paths:
            chapter_number = extract_chapter_number(chapter_path)
            if chapter_number is None:
                continue

            print(f"Processing chapter {chapter_number}...")
            try:
                suggestions = extract_memory_suggestions_for_text(
                    client=client,
                    scene_text=chapter_path.read_text(encoding="utf-8").strip(),
                )
            except Exception as exc:  # Keep terminal app stable for the user.
                print(f"Rebuild failed while processing chapter {chapter_number}: {exc}")
                return

            rebuilt_chapters.append(build_chapter_memory_block(chapter_number, suggestions))
            total_facts += len(suggestions)
            print(f"Extracted {len(suggestions)} fact(s) from chapter {chapter_number}.")

        rebuilt_chapters.sort(key=lambda chapter: chapter["number"])

        try:
            atomic_write(
                CANON_MEMORY_PATH,
                render_canon_memory(rebuilt_chapters),
            )
            log_path = write_rebuild_log(
                mode="FULL",
                lines=[
                    "chapters processed: "
                    + ", ".join(str(chapter["number"]) for chapter in rebuilt_chapters),
                    f"total facts extracted: {total_facts}",
                ],
            )
        except OSError as exc:
            print(f"Rebuild completed, but could not save files: {exc}")
            return

        print()
        print(
            f"Full rebuild complete. Wrote {len(rebuilt_chapters)} chapter block(s) "
            f"and {total_facts} fact(s) to {CANON_MEMORY_PATH}."
        )
        print(f"Rebuild log saved to {log_path}.")
        return

    if selection == "2":
        chapter_number = prompt_for_chapter_number()
        if chapter_number is None:
            return

        chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
        if not chapter_path.exists() or not chapter_path.is_file():
            print(f"Chapter file not found: {chapter_path}")
            return

        print(f"Processing chapter {chapter_number}...")
        try:
            suggestions = extract_memory_suggestions_for_text(
                client=client,
                scene_text=chapter_path.read_text(encoding="utf-8").strip(),
            )
        except Exception as exc:  # Keep terminal app stable for the user.
            print(f"Single-chapter rebuild failed: {exc}")
            return

        rebuilt_chapter = build_chapter_memory_block(chapter_number, suggestions)
        existing_content = (
            CANON_MEMORY_PATH.read_text(encoding="utf-8")
            if CANON_MEMORY_PATH.exists()
            else ""
        )
        chapters = parse_canon_memory(existing_content)
        chapters = insert_or_replace_chapter_block(chapters, rebuilt_chapter)

        try:
            atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))
            log_path = write_rebuild_log(
                mode="SINGLE",
                lines=[
                    f"chapter number: {chapter_number}",
                    f"facts extracted: {len(suggestions)}",
                ],
            )
        except OSError as exc:
            print(f"Single-chapter rebuild completed, but could not save files: {exc}")
            return

        print()
        print(
            f"Single-chapter rebuild complete for chapter {chapter_number}. "
            f"Saved {len(suggestions)} fact(s) to {CANON_MEMORY_PATH}."
        )
        print(f"Rebuild log saved to {log_path}.")
        return

    print("Invalid selection. Enter 1 or 2.")


def handle_rebuild_summaries(client: Any) -> None:
    """Rebuild stored scene summaries from the current chapter files."""
    ensure_project_files()
    print("Rebuild summary options:")
    print()
    print("1) Full novel")
    print("2) Single chapter")

    try:
        selection = input("> ").strip()
    except EOFError:
        print()
        return

    if selection == "1":
        if not CHAPTERS_DIR.exists():
            print(f"Missing chapters directory: {CHAPTERS_DIR}")
            return

        chapter_paths = load_sorted_chapter_paths()
        if not chapter_paths:
            print("No chapter files found.")
            return

        try:
            atomic_write(SCENE_SUMMARIES_PATH, "")
        except OSError as exc:
            print(f"Could not clear scene summaries: {exc}")
            return

        for chapter_path in chapter_paths:
            chapter_number = extract_chapter_number(chapter_path)
            if chapter_number is None:
                continue

            print(f"Processing chapter {chapter_number}...")
            try:
                summary_text = generate_scene_summary_for_chapter(client, chapter_number)
                append_scene_summary(chapter_number, summary_text)
            except FileNotFoundError as exc:
                print(exc)
                continue
            except ValueError as exc:
                print(f"Skipped chapter {chapter_number}: {exc}")
                continue
            except Exception as exc:  # Keep terminal app stable for the user.
                print(f"Rebuild summaries failed while processing chapter {chapter_number}: {exc}")
                return

        print(f"Scene summaries rebuilt at {SCENE_SUMMARIES_PATH}.")
        return

    if selection == "2":
        chapter_number = prompt_for_chapter_number()
        if chapter_number is None:
            return

        chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
        if not chapter_path.exists() or not chapter_path.is_file():
            print(f"Chapter file not found: {chapter_path}")
            return

        try:
            removed_existing = remove_scene_summary_block(chapter_number)
            print(f"Processing chapter {chapter_number}...")
            summary_text = generate_scene_summary_for_chapter(client, chapter_number)
            append_scene_summary(chapter_number, summary_text)
        except ValueError as exc:
            print(f"Could not rebuild chapter {chapter_number}: {exc}")
            return
        except OSError as exc:
            print(f"Could not update scene summaries: {exc}")
            return
        except Exception as exc:  # Keep terminal app stable for the user.
            print(f"Single-chapter summary rebuild failed: {exc}")
            return

        if removed_existing:
            print(f"Rebuilt scene summary for chapter {chapter_number}.")
        else:
            print(
                f"No existing scene summary block found for chapter {chapter_number}. "
                "Added a new one."
            )
        print(f"Scene summaries updated at {SCENE_SUMMARIES_PATH}.")
        return

    print("Invalid selection. Enter 1 or 2.")


def handle_proofread(client: Any) -> None:
    """Proofread pasted text in a fully isolated request."""
    print("Paste text to proofread. Type END on a new line when finished.")
    text_to_proofread = collect_multiline_input(end_marker="END")
    text_to_proofread = clean_terminal_text(text_to_proofread)

    if not text_to_proofread:
        print("No text provided.")
        return

    try:
        result = request_chat_completion(
            client=client,
            messages=build_proofread_messages(text_to_proofread),
            temperature=PROOFREAD_TEMPERATURE,
        )
    except Exception:  # Keep terminal app stable for the user.
        print("Proofread failed.")
        return

    print()
    print(result)




def handle_idea_resurface(client: Any) -> None:
    """Resurface only highly relevant saved ideas for the selected chapter."""
    ensure_project_files()
    chapter_number = prompt_for_chapter_number(prompt_text="Enter chapter number:")
    if chapter_number is None:
        return

    chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
    if not chapter_path.exists() or not chapter_path.is_file():
        print(f"Chapter file not found: {chapter_path}")
        return

    ideas_block = load_ideas_block()
    if not ideas_block:
        print("No ideas available.")
        return

    chapter_text = chapter_path.read_text(encoding="utf-8").strip()
    canon_memory_block = load_memory_block()
    messages = build_idea_resurface_messages(
        chapter_text=chapter_text,
        canon_memory_block=canon_memory_block,
        ideas_block=ideas_block,
    )

    try:
        result = request_chat_completion(
            client=client,
            messages=messages,
            temperature=IDEA_RESURFACE_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Idea resurfacing failed: {exc}")
        return

    print()
    print(result)



def handle_build_book(clean: bool = False) -> None:
    """Compile all numbered chapter files into a manuscript file."""
    if not CHAPTERS_DIR.exists():
        print("No chapters found." if clean else "No chapter files found.")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapters found." if clean else "No chapter files found.")
        return

    manuscript_text = (
        build_clean_manuscript_text(chapter_paths)
        if clean
        else build_manuscript_text(chapter_paths)
    )
    output_path = CLEAN_MANUSCRIPT_PATH if clean else MANUSCRIPT_PATH

    try:
        MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(output_path, manuscript_text)
    except OSError as exc:
        print(f"Manuscript build failed: {exc}")
        return

    print("Clean manuscript built successfully." if clean else "Manuscript built successfully.")



def handle_ideas() -> None:
    """Capture a freeform writing idea without affecting assistant state."""
    print("Paste idea. Type END on a new line when finished.")
    idea_text = collect_multiline_input(end_marker="END")
    idea_text = clean_terminal_text(idea_text)

    if not idea_text:
        print("No idea entered.")
        return

    try:
        append_idea(idea_text)
    except OSError:
        print("Idea could not be saved.")


def handle_world_add() -> None:
    """Capture and save one structured world rule entry."""
    print("Enter world rule category:")
    try:
        category = input("> ").strip()
    except EOFError:
        print()
        return

    if not category:
        print("No category entered.")
        return

    print("Enter rule text:")
    try:
        rule_text = input("> ").strip()
    except EOFError:
        print()
        return

    if not rule_text:
        print("No rule text entered.")
        return

    try:
        append_world_rule(category, rule_text)
    except (OSError, ValueError) as exc:
        print(f"World rule could not be saved: {exc}")
        return

    print(f"World rule saved to {WORLD_RULES_PATH}.")



def handle_export_chapter() -> None:
    """Prepare a chapter text file for manual WordGrinder export."""
    ensure_project_files()
    chapter_number = prompt_for_chapter_number()
    if chapter_number is None:
        return

    export_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
    if export_path.exists() and not prompt_for_confirmation(
        "Overwrite existing export? (y/n)"
    ):
        print("Export cancelled.")
        return

    try:
        atomic_write(export_path, "")
    except OSError as exc:
        print(f"Could not prepare export file: {exc}")
        return

    print()
    print("Now open your WordGrinder chapter and press:")
    print("CTRL+SHIFT+E (export)")
    print("Choose:")
    print("Plain Text")
    print("Save to:")
    print(export_path)
    print()
    print("Press ENTER after exporting the chapter.")

    try:
        input()
    except EOFError:
        print()

    if export_path.exists():
        print("Chapter export complete.")
        return

    print("Chapter export file was not found.")


# ============================================================
# Main application loop
# ============================================================


def print_welcome() -> None:
    """Show a simple startup message."""
    print("Novel AI Assistant")
    print(f"Model: {MODEL_NAME}")
    print(f"Script path: {NOVEL_AI_SCRIPT_PATH}")
    print(f"Project path: {NOVEL_PROJECT_DIR}")
    print("Type /help for commands or type exit to quit.")



def print_help() -> None:
    """Show available commands."""
    print("Available commands:")
    print("  /scene-summary")
    print("  /rebuild-summaries")
    print("  /rebuild-memory")
    print("  /continuity-check")
    print("  /proofread")
    print("  /idea-resurface")
    print("  /build-book")
    print("  /build-book --clean")
    print("  /ideas")
    print("  /world-add")
    print("  /export-chapter")
    print("  /help")
    print("  exit")



def main() -> None:
    """Run the terminal assistant."""
    ensure_project_files()

    try:
        client = create_client()
    except RuntimeError as exc:
        print(exc)
        return

    conversation_history: list[dict[str, str]] = []

    command_handlers: dict[str, Callable[[], None]] = {
        "/scene-summary": lambda: handle_scene_summary(client),
        "/rebuild-summaries": lambda: handle_rebuild_summaries(client),
        "/rebuild-memory": lambda: handle_rebuild_memory(client),
        "/continuity-check": lambda: handle_continuity_check(client),
        "/proofread": lambda: handle_proofread(client),
        "/idea-resurface": lambda: handle_idea_resurface(client),
        "/build-book": handle_build_book,
        "/ideas": handle_ideas,
        "/world-add": handle_world_add,
        "/export-chapter": handle_export_chapter,
        "/help": print_help,
    }

    print_welcome()

    while True:
        user_input = ""
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            print("\nGoodbye.")
            break
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input == "/build-book --clean":
            handle_build_book(clean=True)
            continue

        if user_input in command_handlers:
            command_handlers[user_input]()
            continue

        memory_block = load_memory_block()
        conversation_history = conversation_history[-MAX_CONVERSATION_TURNS * 2:]
        messages = build_main_messages(memory_block, conversation_history, user_input)

        try:
            reply = request_chat_completion(
                client=client,
                messages=messages,
                temperature=MAIN_TEMPERATURE,
            )
        except Exception as exc:  # Keep terminal app stable for the user.
            print(f"Assistant request failed: {exc}")
            continue

        print(f"\nAssistant: {reply}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
