from __future__ import annotations

import os
import re
from collections import OrderedDict
from copy import deepcopy
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
CONTINUITY_REPORTS_DIR = PROJECT_ANALYSIS_DIR / "continuity_reports"
BOOK_INTEGRITY_REPORTS_DIR = PROJECT_ANALYSIS_DIR / "book_integrity_reports"
REBUILD_LOG_DIR = PROJECT_ANALYSIS_DIR / "rebuild_logs"
DRAFTS_DIR = NOVEL_PROJECT_DIR / "drafts"
BACKUPS_DIR = NOVEL_PROJECT_DIR / "backups"
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
DRAFT_PASS_TEMPERATURE = 0.2

CONTINUITY_CHAPTER_WINDOW = 3
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

STORY_STATE_SIGNAL_KEYWORDS = (
    "urgency",
    "urgent",
    "countdown",
    "deadline",
    "danger",
    "threat",
    "risk",
    "unstable",
    "instability",
    "worsening",
    "deteriorating",
    "hidden",
    "secret",
    "withheld",
    "pressure",
    "strain",
    "conflict",
    "uncertainty",
    "hazard",
    "environmental",
    "approaching",
    "mission",
    "failure",
    "collapse",
    "panic",
    "fear",
    "tension",
    "mystery",
    "exposed",
    "running out",
    "time pressure",
)

STORY_STATE_PRIORITY_CATEGORIES = {
    "Injury",
    "Mission State — Active",
    "Mission State — Resolved",
    "Psychological State — Active",
    "Psychological State — Resolved",
    "Relationship State — Active",
    "Relationship State — Resolved",
    "Technology State — Active",
    "Technology State — Resolved",
    "Timeline",
    "Character",
    "Relationship",
    "World",
    "Location",
    "Object",
}

STORY_STATE_HEADING = "[StoryState]"
STORY_STATE_STATUS_PATTERN = re.compile(r"^STATE:\s*(ACTIVE|RESOLVED)\s*$", re.IGNORECASE)
STORY_STATE_FIRST_SEEN_PATTERN = re.compile(r"^FIRST_SEEN:\s*Chapter\s+(\d+)\s*$", re.IGNORECASE)
STORY_STATE_RESOLVED_PATTERN = re.compile(r"^RESOLVED_IN:\s*Chapter\s+(\d+)\s*$", re.IGNORECASE)


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

STORY STATE UPDATES

ACTIVATE
1. Description of unresolved narrative pressure
2. Description of unresolved narrative pressure

RESOLVE
1. Short reference label
2. Short reference label

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
- STORY STATE UPDATES must still be included even if MEMORY SUGGESTIONS is None.

Rules for STORY STATE UPDATES:
- Track cinematic unresolved pressures that should persist across chapters.
- Detect urgency, countdowns, approaching danger, worsening conditions, hidden problems, emotional strain, rising conflict, strategic uncertainty, environmental threat, time pressure, secrecy, withheld information, and mission instability.
- Track both plot-level threats and internal/relationship pressure when they materially affect future narrative movement.
- Under ACTIVATE, list only unresolved conditions that should become persistent Story States.
- Under RESOLVE, list short labels for persistent Story States clearly stabilised, closed, or resolved by this scene.
- If a new pressure is clearly a continuation of an existing state, use wording that closely matches the existing state instead of inventing a separate one.
- Descriptions must be short, concrete, and cinematic.
- If no items apply in a subsection, write exactly: None

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

BOOK_INTEGRITY_SYSTEM_PROMPT = """You are a professional novel structural editor.

Analyse the FULL novel draft.

Return a BOOK INTEGRITY REPORT.

You must evaluate:

STRUCTURE

- pacing imbalance
- weak openings
- rushed climaxes
- chapter length inconsistency

CONTINUITY

- injury continuity errors
- location contradictions
- timeline breaks
- knowledge inconsistencies

CHARACTER ARCS

- unresolved motivations
- inconsistent emotional progression
- sudden unexplained behavioural shifts

PLOT THREAD STATUS

- unresolved setups
- dropped conflicts
- premature resolutions

TENSION FLOW

- flat narrative zones
- spikes without build-up
- missing escalation

WORLD CONSISTENCY

- broken rules
- technology drift
- setting contradictions

Do NOT rewrite text.
Do NOT give story ideas.
Do NOT praise writing.

Return format:

BOOK INTEGRITY REPORT

<sections listed above>"""

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


CHAPTER_SUMMARY_SYSTEM_PROMPT = """You are a narrative summarisation engine for a novel project.

You must analyse ONLY the provided chapter text and canon memory.

Return output in this exact structure:

CHAPTER SUMMARY

Key Events:

- ...
- ...

Character Movement:

- ...
- ...

New World Information:

- ...

Tension Movement:

- Rising
  OR
- Stable
  OR
- Climax
  OR
- Resolution

Unresolved Threads Introduced:

- ...

Resolved Threads:

- ...

STORY STATE UPDATES

ACTIVATE
- ...

RESOLVE
- ...

Rules:

- Be factual.
- Do not give writing advice.
- Do not rewrite scenes.
- Do not praise the writing.
- Do not speculate beyond the text.
- In STORY STATE UPDATES, capture unresolved narrative escalation, psychological pressure, relationship strain, mission risk, environmental threat, secrecy, and strategic instability that should persist across chapters.
- Use ACTIVATE for ongoing pressures and RESOLVE for tensions clearly stabilised or concluded.
- If nothing applies under a subsection, write `- None`."""


DRAFT_PASS_SYSTEM_PROMPT_TEMPLATE = """You are a professional novel developmental editor.

You must analyse the provided text.

You must NOT rewrite any part of the story.
You must NOT give grammar corrections.
You must NOT give publishing advice.
You must NOT invent new plot ideas.

You must ONLY evaluate the requested dimension:

{dimension_instructions}

Return output in this format:

<{dimension_name} PASS>

Strengths:
- bullet points

Weaknesses:
- bullet points

Suggested Improvements:
- bullet points
"""


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
    BOOK_INTEGRITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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


def load_memory_block(*, full: bool = False) -> str:
    """Load canonical story memory for the main assistant and continuity checker."""
    canon_text = read_text_file(CANON_MEMORY_PATH)
    if full:
        return canon_text
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
    current_story_state: dict[str, Any] | None = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped == "========================":
            continue

        chapter_match = CHAPTER_HEADER_PATTERN.fullmatch(stripped)
        if chapter_match is not None:
            current_story_state = None
            current_chapter = {
                "number": int(chapter_match.group(1)),
                "categories": OrderedDict(),
                "story_states": [],
            }
            chapters.append(current_chapter)
            current_category = None
            continue

        if current_story_state is not None:
            if not stripped:
                current_story_state = None
                continue

            if stripped == STORY_STATE_HEADING:
                current_chapter["story_states"].append(
                    {
                        "description": "",
                        "state": "ACTIVE",
                        "first_seen": current_chapter["number"],
                        "resolved_in": None,
                    }
                )
                current_story_state = current_chapter["story_states"][-1]
                continue

            state_match = STORY_STATE_STATUS_PATTERN.fullmatch(stripped)
            if state_match is not None:
                current_story_state["state"] = state_match.group(1).upper()
                continue

            first_seen_match = STORY_STATE_FIRST_SEEN_PATTERN.fullmatch(stripped)
            if first_seen_match is not None:
                current_story_state["first_seen"] = int(first_seen_match.group(1))
                continue

            resolved_match = STORY_STATE_RESOLVED_PATTERN.fullmatch(stripped)
            if resolved_match is not None:
                current_story_state["resolved_in"] = int(resolved_match.group(1))
                continue

            description = current_story_state.get("description", "")
            current_story_state["description"] = (
                f"{description} {stripped}".strip() if description else stripped
            )
            continue

        if not stripped:
            continue

        if stripped == STORY_STATE_HEADING and current_chapter is not None:
            current_story_state = {
                "description": "",
                "state": "ACTIVE",
                "first_seen": current_chapter["number"],
                "resolved_in": None,
            }
            current_chapter["story_states"].append(current_story_state)
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
        for story_state in chapter.get("story_states", []):
            description = str(story_state.get("description", "")).strip()
            if not description:
                continue
            lines.extend(
                [
                    "",
                    STORY_STATE_HEADING,
                    description,
                    f"STATE: {str(story_state.get('state', 'ACTIVE')).upper()}",
                    f"FIRST_SEEN: Chapter {int(story_state.get('first_seen', chapter['number']))}",
                ]
            )
            resolved_in = story_state.get("resolved_in")
            if resolved_in is not None:
                lines.append(f"RESOLVED_IN: Chapter {int(resolved_in)}")
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


def normalize_story_state_description(description: str) -> str:
    """Normalize a story-state description for matching and deduplication."""
    return normalize_fact_text(description)


def story_states_are_similar(state_a: str | dict[str, Any], state_b: str | dict[str, Any]) -> bool:
    """Return True when two story-state descriptions represent the same persistent pressure."""
    text_a = (
        str(state_a.get("description", "")).strip()
        if isinstance(state_a, dict)
        else str(state_a).strip()
    )
    text_b = (
        str(state_b.get("description", "")).strip()
        if isinstance(state_b, dict)
        else str(state_b).strip()
    )
    return facts_are_similar(text_a, text_b)


def should_track_story_state(fact_text: str, category: str) -> bool:
    """Return True when a canon fact should also become a persistent cinematic Story State."""
    cleaned_fact = normalize_story_state_description(fact_text)
    if not cleaned_fact:
        return False

    if category in {
        "Mission State — Active",
        "Mission State — Resolved",
        "Psychological State — Active",
        "Psychological State — Resolved",
        "Relationship State — Active",
        "Relationship State — Resolved",
        "Technology State — Active",
        "Technology State — Resolved",
        "Injury",
    }:
        return True

    if category not in STORY_STATE_PRIORITY_CATEGORIES:
        return False

    return any(keyword in cleaned_fact for keyword in STORY_STATE_SIGNAL_KEYWORDS)


def iter_story_states(chapters: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return chapter/state pairs for every parsed Story State."""
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for chapter in chapters:
        for story_state in chapter.get("story_states", []):
            pairs.append((chapter, story_state))
    return pairs


def add_or_update_story_state(
    chapters: list[dict[str, Any]],
    description: str,
    first_seen: int,
    state: str = "ACTIVE",
    resolved_in: int | None = None,
) -> bool:
    """Create or update one persistent Story State without duplicating continuations."""
    cleaned_description = description.strip()
    normalized_state = "RESOLVED" if state.strip().upper() == "RESOLVED" else "ACTIVE"
    if not cleaned_description:
        return False

    matched_chapter: dict[str, Any] | None = None
    matched_story_state: dict[str, Any] | None = None

    for chapter, story_state in iter_story_states(chapters):
        if not story_states_are_similar(story_state, cleaned_description):
            continue
        matched_chapter = chapter
        matched_story_state = story_state
        break

    if matched_story_state is not None and matched_chapter is not None:
        existing_description = str(matched_story_state.get("description", "")).strip()
        if len(cleaned_description) > len(existing_description):
            matched_story_state["description"] = cleaned_description
        matched_story_state["first_seen"] = min(
            int(matched_story_state.get("first_seen", first_seen)),
            first_seen,
        )
        if normalized_state == "RESOLVED":
            matched_story_state["state"] = "RESOLVED"
            matched_story_state["resolved_in"] = resolved_in or first_seen
        else:
            matched_story_state["state"] = "ACTIVE"
            matched_story_state["resolved_in"] = None
        return True

    target_chapter: dict[str, Any] | None = None
    for chapter in chapters:
        if chapter["number"] == first_seen:
            target_chapter = chapter
            break

    if target_chapter is None:
        target_chapter = {
            "number": first_seen,
            "categories": OrderedDict(),
            "story_states": [],
        }
        chapters.append(target_chapter)
        chapters.sort(key=lambda chapter: chapter["number"])

    target_chapter.setdefault("story_states", []).append(
        {
            "description": cleaned_description,
            "state": normalized_state,
            "first_seen": first_seen,
            "resolved_in": resolved_in if normalized_state == "RESOLVED" else None,
        }
    )
    return True


def sync_story_states_from_suggestions(
    chapters: list[dict[str, Any]],
    chapter_number: int,
    suggestions: list[tuple[int, str, str]],
) -> int:
    """Derive Story States from canon suggestions and update the canon-memory model."""
    updates = 0
    for _, fact, category in suggestions:
        if not should_track_story_state(fact, category):
            continue

        state = "ACTIVE"
        resolved_in: int | None = None
        if category in ACTIVE_TO_RESOLVED_CATEGORY.values():
            state = "RESOLVED"
            resolved_in = chapter_number

        if add_or_update_story_state(
            chapters,
            description=fact,
            first_seen=chapter_number,
            state=state,
            resolved_in=resolved_in,
        ):
            updates += 1

    return updates


def consolidate_story_states(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate Story States across the whole novel while preserving earliest origin."""
    consolidated_chapters = deepcopy(chapters)
    original_states: list[dict[str, Any]] = []

    for chapter in consolidated_chapters:
        original_states.extend(
            [
                {
                    "description": str(state.get("description", "")).strip(),
                    "state": str(state.get("state", "ACTIVE")).upper(),
                    "first_seen": int(state.get("first_seen", chapter["number"])),
                    "resolved_in": state.get("resolved_in"),
                }
                for state in chapter.get("story_states", [])
                if str(state.get("description", "")).strip()
            ]
        )
        chapter["story_states"] = []

    for story_state in sorted(original_states, key=lambda state: state["first_seen"]):
        add_or_update_story_state(
            consolidated_chapters,
            description=story_state["description"],
            first_seen=story_state["first_seen"],
            state=story_state["state"],
            resolved_in=story_state["resolved_in"],
        )

    consolidated_chapters.sort(key=lambda chapter: chapter["number"])
    return consolidated_chapters


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
        target_chapter = {
            "number": chapter_number,
            "categories": OrderedDict(),
            "story_states": [],
        }
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
    story_state_updates = sync_story_states_from_suggestions(chapters, chapter_number, [
        (index, fact, category) for index, (fact, category) in enumerate(cleaned_facts, start=1)
    ])
    if saved_count > 0 or story_state_updates > 0:
        chapters = consolidate_story_states(chapters)
        atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))

    print(f"Saved {saved_count} canon fact(s).")
    print(f"Skipped {skipped_duplicates} duplicate fact(s).")
    if story_state_updates > 0:
        print(f"Updated {story_state_updates} story state(s).")



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

    chapter_block = {
        "number": chapter_number,
        "categories": order_memory_categories(categories),
        "story_states": [],
    }
    sync_story_states_from_suggestions([chapter_block], chapter_number, suggestions)
    return chapter_block



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


def chunk_text_blocks(blocks: list[str], max_chars: int = 12000) -> list[str]:
    """Group text blocks into bounded chunks for safer long-form analysis calls."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue

        block_length = len(block)
        separator_length = 2 if current_parts else 0

        if current_parts and current_length + separator_length + block_length > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [block]
            current_length = block_length
            continue

        if block_length > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0
            start = 0
            while start < block_length:
                end = min(start + max_chars, block_length)
                chunks.append(block[start:end])
                start = end
            continue

        current_parts.append(block)
        current_length += separator_length + block_length

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def warn_for_missing_chapter_files(chapter_paths: list[Path]) -> None:
    """Print warnings for gaps in numbered chapter files without stopping execution."""
    chapter_numbers = [
        extract_chapter_number(path)
        for path in chapter_paths
    ]
    ordered_numbers = sorted(number for number in chapter_numbers if number is not None)
    if not ordered_numbers:
        return

    expected = set(range(ordered_numbers[0], ordered_numbers[-1] + 1))
    missing_numbers = sorted(expected - set(ordered_numbers))
    for missing_number in missing_numbers:
        print(f"Warning: Missing chapter_{missing_number}.txt")


def prompt_for_destructive_confirmation() -> bool:
    """Require an explicit yes confirmation before destructive operations."""
    print("This will overwrite current project state. Continue? (yes/no)")
    try:
        response = input("> ").strip()
    except EOFError:
        print()
        return False
    return response == "yes"


def create_canon_memory_backup() -> None:
    """Create a timestamped canon memory backup before rebuild operations."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    backup_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUPS_DIR / f"canon_memory_backup_{backup_timestamp}.txt"
    backup_content = (
        CANON_MEMORY_PATH.read_text(encoding="utf-8")
        if CANON_MEMORY_PATH.exists()
        else ""
    )
    atomic_write(backup_path, backup_content)
    print("Canon memory backup created.")



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


def parse_story_state_updates(result_text: str) -> tuple[list[str], list[str]]:
    """Parse STORY STATE UPDATES sections into activation and resolution labels."""
    match = re.search(
        r"STORY STATE UPDATES\s*(.*?)(?:\n[A-Z][A-Z ]+\n|\Z)",
        result_text,
        re.DOTALL,
    )
    if match is None:
        return [], []

    section_text = match.group(1).strip()
    activate_match = re.search(
        r"ACTIVATE\s*(.*?)(?:\nRESOLVE\s|\Z)",
        section_text,
        re.DOTALL,
    )
    resolve_match = re.search(r"RESOLVE\s*(.*)\Z", section_text, re.DOTALL)

    def parse_items(raw_text: str) -> list[str]:
        cleaned = raw_text.strip()
        if not cleaned or cleaned.lower() == "none":
            return []

        items: list[str] = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            item_match = re.match(r"^(?:\d+\.|-)\s*(.+?)\s*$", stripped)
            if item_match is None:
                continue
            item_text = item_match.group(1).strip()
            if item_text.lower() == "none":
                return []
            items.append(item_text)
        return items

    activated = parse_items(activate_match.group(1) if activate_match else "")
    resolved = parse_items(resolve_match.group(1) if resolve_match else "")
    return activated, resolved



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



def apply_resolutions(selected_labels: list[str], chapter_number: int | None = None) -> int:
    """Move matching ACTIVE canon facts and Story States into their resolved lifecycle state."""
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
        story_state_resolved = False

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
            for _, story_state in iter_story_states(chapters):
                if (
                    str(story_state.get("state", "ACTIVE")).upper() == "ACTIVE"
                    and resolution_label_matches_fact(label, str(story_state.get("description", "")))
                ):
                    story_state["state"] = "RESOLVED"
                    story_state["resolved_in"] = chapter_number
                    resolved_count += 1
                    story_state_resolved = True
                    break
            if matched_entry is None:
                if story_state_resolved:
                    continue
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
        chapters = consolidate_story_states(chapters)
        atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))

    return resolved_count


def apply_story_state_updates(
    chapter_number: int,
    activated_states: list[str],
    resolved_labels: list[str],
) -> tuple[int, int]:
    """Persist Story State activations and resolutions from an analysis response."""
    ensure_project_files()
    chapters = parse_canon_memory(CANON_MEMORY_PATH.read_text(encoding="utf-8"))
    activations_applied = 0

    for description in activated_states:
        if add_or_update_story_state(
            chapters,
            description=description,
            first_seen=chapter_number,
            state="ACTIVE",
        ):
            activations_applied += 1

    resolutions_applied = 0
    for label in resolved_labels:
        for _, story_state in iter_story_states(chapters):
            if (
                str(story_state.get("state", "ACTIVE")).upper() == "ACTIVE"
                and resolution_label_matches_fact(label, str(story_state.get("description", "")))
            ):
                story_state["state"] = "RESOLVED"
                story_state["resolved_in"] = chapter_number
                resolutions_applied += 1
                break

    if activations_applied > 0 or resolutions_applied > 0:
        chapters = consolidate_story_states(chapters)
        atomic_write(CANON_MEMORY_PATH, render_canon_memory(chapters))

    return activations_applied, resolutions_applied


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


def build_book_integrity_messages(
    full_novel_block: str,
    canon_memory_block: str,
    world_rules_block: str,
) -> list[dict[str, str]]:
    """Build the isolated message list for full-book integrity analysis."""
    return [
        {
            "role": "system",
            "content": BOOK_INTEGRITY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Full novel draft:\n\n{full_novel_block}\n\n"
                f"Canon memory:\n\n{canon_memory_block}\n\n"
                f"World rules:\n\n{world_rules_block}"
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


def build_chapter_summary_messages(
    chapter_text: str,
    canon_memory_block: str,
) -> list[dict[str, str]]:
    """Build the message list for chapter-level narrative summarisation."""
    return [
        {
            "role": "system",
            "content": CHAPTER_SUMMARY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Chapter text:\n\n{chapter_text}\n\n"
                f"Canon memory:\n\n{canon_memory_block}"
            ),
        },
    ]


def build_draft_pass_messages(
    text_to_analyse: str,
    dimension_name: str,
    dimension_instructions: str,
) -> list[dict[str, str]]:
    """Build the isolated message list for draft analysis passes."""
    system_prompt = DRAFT_PASS_SYSTEM_PROMPT_TEMPLATE.format(
        dimension_name=dimension_name,
        dimension_instructions=dimension_instructions,
    )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": text_to_analyse,
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

    activated_states, resolved_story_states = parse_story_state_updates(result)
    try:
        activated_count, resolved_story_count = apply_story_state_updates(
            chapter_number=chapter_number,
            activated_states=activated_states,
            resolved_labels=resolved_story_states,
        )
    except OSError as exc:
        print(f"Could not update story states: {exc}")
        return

    if activated_count > 0 or resolved_story_count > 0:
        print(
            f"\nStory States updated: {activated_count} activated, "
            f"{resolved_story_count} resolved."
        )

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
                resolved_count = apply_resolutions(selected_labels, chapter_number=chapter_number)
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

    selected_index = chapter_paths.index(selected_path)
    previous_paths = chapter_paths[
        max(0, selected_index - CONTINUITY_CHAPTER_WINDOW):selected_index
    ]
    print(
        f"Using last {len(previous_paths)} chapters for continuity context."
    )

    memory_block = load_memory_block(full=True)
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



def handle_book_integrity(client: Any) -> None:
    """Run a full-novel structural and continuity integrity audit."""
    ensure_project_files()

    if not CHAPTERS_DIR.exists():
        print(f"Missing chapters directory: {CHAPTERS_DIR}")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found in ~/writing/novel_project/chapters/")
        return

    chapter_blocks: list[str] = []
    for chapter_path in chapter_paths:
        chapter_number = extract_chapter_number(chapter_path)
        if chapter_number is None:
            continue
        chapter_text = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
        chapter_blocks.append(f"CHAPTER {chapter_number}\n\n{chapter_text}")

    chunked_blocks = chunk_text_blocks(chapter_blocks)
    if not chunked_blocks:
        print("No chapter text found.")
        return

    canon_memory_block = load_memory_block(full=True)
    world_rules_block = (
        read_text_file(WORLD_RULES_PATH)
        if WORLD_RULES_PATH.exists()
        else "(not provided)"
    )

    try:
        chunk_reports: list[str] = []
        for chunk_index, chunk_text in enumerate(chunked_blocks, start=1):
            print(f"Analyzing chunk {chunk_index}/{len(chunked_blocks)}...")
            chunk_report = request_chat_completion(
                client=client,
                messages=build_book_integrity_messages(
                    full_novel_block=chunk_text,
                    canon_memory_block=canon_memory_block,
                    world_rules_block=world_rules_block,
                ),
                temperature=CONTINUITY_TEMPERATURE,
            )
            chunk_reports.append(chunk_report)

        if len(chunk_reports) == 1:
            report = chunk_reports[0]
        else:
            merge_messages = build_book_integrity_messages(
                full_novel_block=(
                    "Merge these per-chunk book-integrity analyses into one final report. "
                    "Preserve concrete issues, remove duplicates, and keep recommendations actionable.\n\n"
                    + "\n\n".join(
                        f"Chunk {index} report:\n{chunk_report}"
                        for index, chunk_report in enumerate(chunk_reports, start=1)
                    )
                ),
                canon_memory_block=canon_memory_block,
                world_rules_block=world_rules_block,
            )
            report = request_chat_completion(
                client=client,
                messages=merge_messages,
                temperature=CONTINUITY_TEMPERATURE,
            )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Book integrity analysis failed: {exc}")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    report_path = BOOK_INTEGRITY_REPORTS_DIR / f"book_integrity_{timestamp}.txt"
    try:
        atomic_write(report_path, report + "\n")
    except OSError as exc:
        print(f"Book integrity report could not be saved: {exc}")
        return

    print()
    print(report)
    print()
    print(f"Book integrity report saved to {report_path}.")


def handle_rebuild_memory(client: Any, command_text: str = "") -> None:
    """Rebuild canon memory for the whole novel or one chapter."""
    ensure_project_files()
    normalized_command = command_text.strip().lower()
    forced_full = normalized_command.startswith("/rebuild-memory full")
    forced_single = normalized_command.startswith("/rebuild-memory single")

    selection = ""
    if forced_full:
        selection = "1"
    elif forced_single:
        selection = "2"
    else:
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
        if not prompt_for_destructive_confirmation():
            print("Rebuild aborted.")
            return
        try:
            create_canon_memory_backup()
        except OSError as exc:
            print(f"Could not create canon memory backup: {exc}")
            return

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
        rebuilt_chapters = consolidate_story_states(rebuilt_chapters)

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
        try:
            create_canon_memory_backup()
        except OSError as exc:
            print(f"Could not create canon memory backup: {exc}")
            return

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
        chapters = consolidate_story_states(chapters)

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



def handle_chapter_summary(client: Any) -> None:
    """Summarise one full chapter using canon memory for narrative context."""
    ensure_project_files()
    chapter_number = prompt_for_chapter_number(prompt_text="Enter chapter number:")
    if chapter_number is None:
        return

    chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
    if not chapter_path.exists() or not chapter_path.is_file():
        print(f"Chapter file not found: {chapter_path}")
        return

    chapter_text = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
    canon_memory_block = read_text_file(CANON_MEMORY_PATH)

    try:
        result = request_chat_completion(
            client=client,
            messages=build_chapter_summary_messages(
                chapter_text=chapter_text,
                canon_memory_block=canon_memory_block,
            ),
            temperature=SCENE_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Chapter summary failed: {exc}")
        return

    print()
    print(result)

    activated_states, resolved_story_states = parse_story_state_updates(result)
    try:
        activated_count, resolved_story_count = apply_story_state_updates(
            chapter_number=chapter_number,
            activated_states=activated_states,
            resolved_labels=resolved_story_states,
        )
    except OSError as exc:
        print(f"Could not update story states: {exc}")
        return

    if activated_count > 0 or resolved_story_count > 0:
        print(
            f"\nStory States updated: {activated_count} activated, "
            f"{resolved_story_count} resolved."
        )


def handle_draft_pass(client: Any, command_text: str = "") -> None:
    """Run a structured editorial analysis pass without modifying project files."""
    normalized_command = command_text.strip().lower()
    flag_to_selection = {
        "--structure": "1",
        "--tension": "2",
        "--character": "3",
        "--clarity": "4",
    }
    preselected_mode = next(
        (selection for flag, selection in flag_to_selection.items() if flag in normalized_command),
        "",
    )

    if not preselected_mode:
        print("Select analysis mode:")
        print()
        print("1 — Structure")
        print("2 — Tension")
        print("3 — Character")
        print("4 — Clarity")

        try:
            mode_selection = input("> ").strip()
        except EOFError:
            print()
            return
    else:
        mode_selection = preselected_mode

    dimension_map = {
        "1": (
            "STRUCTURE",
            "STRUCTURE:\n"
            "- pacing\n"
            "- scene purpose\n"
            "- narrative momentum\n"
            "- chapter openings\n"
            "- chapter endings\n"
            "- exposition balance",
        ),
        "2": (
            "TENSION",
            "TENSION:\n"
            "- stakes clarity\n"
            "- urgency\n"
            "- mystery drive\n"
            "- threat escalation\n"
            "- emotional pressure",
        ),
        "3": (
            "CHARACTER",
            "CHARACTER:\n"
            "- motivation clarity\n"
            "- emotional realism\n"
            "- dialogue authenticity\n"
            "- relationship dynamics\n"
            "- behavioural consistency",
        ),
        "4": (
            "CLARITY",
            "CLARITY:\n"
            "- readability\n"
            "- sentence density\n"
            "- description overload\n"
            "- technical confusion\n"
            "- flow interruptions",
        ),
    }

    selected_dimension = dimension_map.get(mode_selection)
    if selected_dimension is None:
        print("Invalid selection. Enter 1, 2, 3, or 4.")
        return

    dimension_name, dimension_instructions = selected_dimension

    print()
    print("Run on:")
    print()
    print("1 — Single chapter")
    print("2 — Full novel")

    try:
        scope_selection = input("> ").strip()
    except EOFError:
        print()
        return

    text_to_analyse = ""
    chapter_chunks: list[str] = []
    if scope_selection == "1":
        print("Enter chapter number:")
        try:
            chapter_value = input("> ").strip()
        except EOFError:
            print()
            return

        if not chapter_value or not chapter_value.isdigit() or int(chapter_value) <= 0:
            print("Chapter number must be a positive integer.")
            return

        chapter_number = int(chapter_value)
        chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
        if not chapter_path.exists() or not chapter_path.is_file():
            print(f"Chapter file not found: {chapter_path}")
            return

        text_to_analyse = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
    elif scope_selection == "2":
        if not CHAPTERS_DIR.exists() or not CHAPTERS_DIR.is_dir():
            print(f"Missing chapters directory: {CHAPTERS_DIR}")
            return

        chapter_paths = load_sorted_chapter_paths()
        if not chapter_paths:
            print("No chapter files found.")
            return

        chapter_blocks = []
        for chapter_path in chapter_paths:
            chapter_number = extract_chapter_number(chapter_path)
            if chapter_number is None:
                continue
            chapter_text = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
            chapter_blocks.append(f"CHAPTER {chapter_number}\n\n{chapter_text}")

        chapter_chunks = chunk_text_blocks(chapter_blocks)
        if not chapter_chunks:
            print("No chapter text found.")
            return
        text_to_analyse = "\n\n".join(chapter_chunks).strip()
    else:
        print("Invalid selection. Enter 1 or 2.")
        return

    if not text_to_analyse:
        print("No text found to analyse.")
        return

    try:
        if scope_selection == "2":
            chunk_results: list[str] = []
            for chunk_index, chunk_text in enumerate(chapter_chunks, start=1):
                print(f"Analysing chunk {chunk_index}/{len(chapter_chunks)}...")
                chunk_results.append(
                    request_chat_completion(
                        client=client,
                        messages=build_draft_pass_messages(
                            text_to_analyse=chunk_text,
                            dimension_name=dimension_name,
                            dimension_instructions=dimension_instructions,
                        ),
                        temperature=DRAFT_PASS_TEMPERATURE,
                    )
                )

            if len(chunk_results) == 1:
                result = chunk_results[0]
            else:
                merge_prompt = (
                    "Merge the following draft-pass chunk analyses into one final report. "
                    "Keep duplicates removed, preserve specific issues, and output one coherent pass report.\n\n"
                    + "\n\n".join(
                        f"Chunk {index} analysis:\n{chunk_result}"
                        for index, chunk_result in enumerate(chunk_results, start=1)
                    )
                )
                result = request_chat_completion(
                    client=client,
                    messages=build_draft_pass_messages(
                        text_to_analyse=merge_prompt,
                        dimension_name=dimension_name,
                        dimension_instructions=dimension_instructions,
                    ),
                    temperature=DRAFT_PASS_TEMPERATURE,
                )
        else:
            result = request_chat_completion(
                client=client,
                messages=build_draft_pass_messages(
                    text_to_analyse=text_to_analyse,
                    dimension_name=dimension_name,
                    dimension_instructions=dimension_instructions,
                ),
                temperature=DRAFT_PASS_TEMPERATURE,
            )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Draft pass failed: {exc}")
        return

    print()
    print(result)


def handle_build_book() -> None:
    """Compile all numbered chapter files into a manuscript file."""
    if not CHAPTERS_DIR.exists():
        print("No chapter files found.")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found.")
        return
    warn_for_missing_chapter_files(chapter_paths)

    manuscript_text = build_manuscript_text(chapter_paths)

    try:
        MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(MANUSCRIPT_PATH, manuscript_text)
    except OSError as exc:
        print(f"Manuscript build failed: {exc}")
        return

    print("Manuscript built successfully.")



def handle_draft_save() -> None:
    """Snapshot all chapters into a timestamped draft manuscript."""
    if not CHAPTERS_DIR.exists():
        print("No chapter files found.")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found.")
        return

    draft_text = build_manuscript_text(chapter_paths)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    draft_path = DRAFTS_DIR / f"draft_{timestamp}.txt"

    try:
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        if draft_path.exists():
            print("Draft snapshot already exists for this minute. Try again in a moment.")
            return
        atomic_write(draft_path, draft_text)
    except OSError as exc:
        print(f"Draft snapshot failed: {exc}")
        return

    print("Draft snapshot saved.")


def load_sorted_draft_paths() -> list[Path]:
    """Return saved draft files sorted by filename."""
    if not DRAFTS_DIR.exists():
        return []

    draft_paths: list[Path] = []
    for path in DRAFTS_DIR.iterdir():
        if path.is_file() and path.name.startswith("draft_") and path.suffix == ".txt":
            draft_paths.append(path)
    return sorted(draft_paths, key=lambda path: path.name)


def handle_draft_list() -> list[Path]:
    """Print a numbered list of draft files and return them."""
    draft_paths = load_sorted_draft_paths()
    if not draft_paths:
        print("No drafts found.")
        return []

    for index, path in enumerate(draft_paths, start=1):
        print(f"{index}. {path.name}")
    return draft_paths


def split_manuscript_into_chapters(text: str) -> list[tuple[int, str]]:
    """Split manuscript text into numbered chapter bodies using CHAPTER headers."""
    lines = text.splitlines()
    chapters: list[tuple[int, str]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    def commit_current() -> None:
        nonlocal current_number, current_lines
        if current_number is None:
            return
        chapter_text = "\n".join(current_lines).strip()
        chapters.append((current_number, chapter_text))
        current_number = None
        current_lines = []

    line_index = 0
    while line_index < len(lines):
        stripped = lines[line_index].strip()

        if stripped == "========================" and line_index + 1 < len(lines):
            chapter_match = CHAPTER_HEADER_PATTERN.fullmatch(lines[line_index + 1].strip())
            if chapter_match is not None:
                commit_current()
                current_number = int(chapter_match.group(1))
                line_index += 2
                if (
                    line_index < len(lines)
                    and lines[line_index].strip() == "========================"
                ):
                    line_index += 1
                if line_index < len(lines) and not lines[line_index].strip():
                    line_index += 1
                continue

        chapter_match = CHAPTER_HEADER_PATTERN.fullmatch(stripped)
        if chapter_match is not None:
            commit_current()
            current_number = int(chapter_match.group(1))
            line_index += 1
            if line_index < len(lines) and not lines[line_index].strip():
                line_index += 1
            continue

        if current_number is not None:
            current_lines.append(lines[line_index])
        line_index += 1

    commit_current()
    return chapters


def handle_draft_load() -> None:
    """Restore chapter files from a selected draft snapshot."""
    draft_paths = handle_draft_list()
    if not draft_paths:
        return

    if not prompt_for_destructive_confirmation():
        print("Draft load aborted.")
        return

    print("Choose draft number:")
    try:
        choice = input("> ").strip()
    except EOFError:
        print()
        return

    if not choice.isdigit():
        print("Invalid draft selection.")
        return

    selected_index = int(choice) - 1
    if selected_index < 0 or selected_index >= len(draft_paths):
        print("Invalid draft selection.")
        return

    selected_draft = draft_paths[selected_index]
    try:
        draft_text = selected_draft.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Could not read draft: {exc}")
        return

    chapter_entries = split_manuscript_into_chapters(draft_text)
    if not chapter_entries:
        print("Draft could not be restored: no chapter markers found.")
        return

    try:
        CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        for chapter_number, chapter_text in chapter_entries:
            chapter_path = CHAPTERS_DIR / f"chapter_{chapter_number}.txt"
            atomic_write(chapter_path, chapter_text + ("\n" if chapter_text else ""))
    except OSError as exc:
        print(f"Draft restore failed: {exc}")
        return

    print("Draft restored successfully.")


def handle_save_draft() -> None:
    """Legacy alias for /draft-save."""
    handle_draft_save()


def handle_list_drafts() -> list[Path]:
    """Legacy alias for /draft-list."""
    return handle_draft_list()


def handle_restore_draft() -> None:
    """Legacy alias for /draft-load."""
    handle_draft_load()


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


def handle_novel_stats() -> None:
    """Print chapter and wordcount telemetry for the current novel project."""
    ensure_project_files()
    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found.")
        return

    chapter_wordcounts: list[tuple[str, int]] = []
    total_words = 0
    for chapter_path in chapter_paths:
        chapter_text = clean_terminal_text(chapter_path.read_text(encoding="utf-8"))
        wordcount = len(chapter_text.split())
        chapter_wordcounts.append((chapter_path.name, wordcount))
        total_words += wordcount

    total_chapters = len(chapter_wordcounts)
    average_words = total_words / total_chapters if total_chapters else 0.0
    longest_chapter = max(chapter_wordcounts, key=lambda item: item[1])
    shortest_chapter = min(chapter_wordcounts, key=lambda item: item[1])

    print(f"Total chapters: {total_chapters}")
    print(f"Total wordcount: {total_words}")
    print(f"Average words per chapter: {average_words:.2f}")
    print(f"Longest chapter: {longest_chapter[0]} ({longest_chapter[1]} words)")
    print(f"Shortest chapter: {shortest_chapter[0]} ({shortest_chapter[1]} words)")


TIMELINE_STOPWORDS = {
    "a", "an", "and", "approaches", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "later",
    "of", "on", "or", "the", "their", "them", "to", "toward", "with",
    "active", "resolved", "increase", "increases", "increased", "rises", "rose",
    "rising", "shows", "showing", "signs", "becomes", "becoming", "create",
    "creates", "created", "indicates", "indicating", "reveals", "revealing",
    "marks", "marked", "set", "sets", "setting", "stage",
}


ENVIRONMENT_PLOT_KEYWORDS = {
    "alarm", "anomaly", "breach", "collapse", "countdown", "danger", "debris",
    "environment", "evacuation", "hazard", "impact", "lockdown", "radiation",
    "route", "safe", "scan", "sensor", "shelter", "signal", "storm", "survival",
    "threat", "window",
}


DISCOVERY_KEYWORDS = {
    "discover", "discovery", "find", "found", "identify", "identified", "learn",
    "learns", "realise", "realises", "realize", "realizes", "reveal", "reveals",
    "revealed", "truth", "clue", "evidence", "decode", "decoded", "message",
    "signal", "secret", "unknown", "uncover", "understand", "insight",
}


CONFLICT_KEYWORDS = {
    "argument", "attack", "betrayal", "clash", "collision", "conflict", "crisis",
    "danger", "fear", "fight", "fracture", "hostile", "hostility", "opposes",
    "alarm", "radiation",
    "pressure", "risk", "rupture", "strain", "standoff", "struggle", "tension",
    "threat", "urgent", "urgency", "warning",
}


MISSION_KEYWORDS = {
    "approach", "arrival", "countdown", "crew", "deadline", "deploy", "deployment",
    "engine", "escape", "evacuate", "journey", "launch", "mission", "objective",
    "plan", "preparation", "prepare", "prepared", "progress", "ready", "readiness",
    "reactor", "repair", "response", "route", "schedule", "shield", "ship",
    "survival", "system", "window",
}


CHARACTER_KEYWORDS = {
    "afraid", "doubt", "hesitation", "injury", "pain", "panic", "secret", "shame",
    "sick", "weakness", "wound",
}


RELATIONSHIP_KEYWORDS = {
    "alliance", "blame", "bond", "crew", "distrust", "loyalty", "promise", "responsibility",
    "rift", "team", "trust",
}


def tokenize_timeline_text(text: str) -> set[str]:
    """Return a compact set of meaningful tokens for timeline thread matching."""
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 3 and token not in TIMELINE_STOPWORDS
    }



def is_plot_relevant_event(category: str, text: str) -> bool:
    """Filter out low-impact atmosphere notes from the timeline view."""
    if category not in {"World", "Location"}:
        return True
    tokens = tokenize_timeline_text(text)
    return (
        bool(tokens & ENVIRONMENT_PLOT_KEYWORDS)
        or bool(tokens & DISCOVERY_KEYWORDS)
        or bool(tokens & CONFLICT_KEYWORDS)
    )



def infer_timeline_category(category: str, text: str, state: str) -> str:
    """Map canon-memory categories into the narrative timeline categories."""
    tokens = tokenize_timeline_text(text)

    if category in {
        "Relationship",
        "Relationship State — Active",
        "Relationship State — Resolved",
    }:
        return "Relationship"
    if category in {
        "Character",
        "Injury",
        "Psychological State — Active",
        "Psychological State — Resolved",
    }:
        if tokens & RELATIONSHIP_KEYWORDS:
            return "Relationship"
        return "Character"
    if category in {
        "Mission State — Active",
        "Mission State — Resolved",
        "Technology State — Active",
        "Technology State — Resolved",
    }:
        if tokens & CONFLICT_KEYWORDS and state == "ACTIVE":
            return "Conflict"
        return "Mission"
    if category in {"World", "Location"}:
        if tokens & DISCOVERY_KEYWORDS:
            return "Discovery"
        if tokens & CONFLICT_KEYWORDS:
            return "Conflict"
        return "Environment"
    if category == "Object":
        if tokens & DISCOVERY_KEYWORDS:
            return "Discovery"
        if tokens & MISSION_KEYWORDS:
            return "Mission"
        return "Environment"

    if tokens & DISCOVERY_KEYWORDS:
        return "Discovery"
    if tokens & CONFLICT_KEYWORDS:
        return "Conflict"
    if tokens & CHARACTER_KEYWORDS:
        return "Character"
    if tokens & RELATIONSHIP_KEYWORDS:
        return "Relationship"
    if tokens & MISSION_KEYWORDS or category == "Timeline":
        return "Mission"
    return "Environment"



def build_timeline_event(
    chapter_number: int,
    source_category: str,
    fact: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert one canon-memory fact into a narrative timeline event."""
    text = get_fact_text(fact)
    state = str(fact.get("state", "ACTIVE")).upper()
    if not text or not is_plot_relevant_event(source_category, text):
        return None

    tokens = tokenize_timeline_text(text)
    return {
        "chapter": chapter_number,
        "source_category": source_category,
        "text": text,
        "state": "RESOLVED" if state == "RESOLVED" else "ACTIVE",
        "narrative_category": infer_timeline_category(source_category, text, state),
        "tokens": tokens,
    }



def events_share_thread(event_a: dict[str, Any], event_b: dict[str, Any]) -> bool:
    """Return True when two timeline events likely describe the same narrative thread."""
    if facts_are_similar(event_a["text"], event_b["text"]):
        return True

    shared_tokens = event_a["tokens"] & event_b["tokens"]
    if len(shared_tokens) >= 2:
        return True

    if len(shared_tokens) == 1:
        shared_token = next(iter(shared_tokens))
        if (
            shared_token in {"shield", "engine", "mission", "signal", "reactor"}
            and (
                event_a["narrative_category"] == event_b["narrative_category"]
                or "RESOLVED" in {event_a["state"], event_b["state"]}
            )
        ):
            return True

    return False



def merge_timeline_events(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate timeline threads into a cleaner chronological event list."""
    merged_events: list[dict[str, Any]] = []
    seen_threads: list[dict[str, Any]] = []

    prioritized_categories = [
        "Timeline",
        "Mission State — Active",
        "Mission State — Resolved",
        "Technology State — Active",
        "Technology State — Resolved",
        "Psychological State — Active",
        "Psychological State — Resolved",
        "Relationship State — Active",
        "Relationship State — Resolved",
        "Character",
        "Relationship",
        "Injury",
        "Foreshadowing Setup",
        "Foreshadowing Payoff",
        "Object",
        "World",
        "Location",
    ]

    for chapter in sorted(chapters, key=lambda item: item["number"]):
        chapter_categories = chapter["categories"]
        ordered_category_names = [
            name for name in prioritized_categories if name in chapter_categories
        ]
        ordered_category_names.extend(
            name for name in chapter_categories if name not in ordered_category_names
        )

        for category_name in ordered_category_names:
            for fact in chapter_categories.get(category_name, []):
                event = build_timeline_event(chapter["number"], category_name, fact)
                if event is None:
                    continue

                existing_thread = next(
                    (thread for thread in seen_threads if events_share_thread(thread, event)),
                    None,
                )

                if existing_thread is None:
                    thread_event = dict(event)
                    seen_threads.append(thread_event)
                    merged_events.append(event)
                    continue

                if event["state"] == existing_thread["state"]:
                    existing_thread["tokens"] = existing_thread["tokens"] | event["tokens"]
                    if event["narrative_category"] in {"Conflict", "Discovery"}:
                        existing_thread["narrative_category"] = event["narrative_category"]
                    if len(event["text"]) > len(existing_thread["text"]):
                        existing_thread["text"] = event["text"]
                    continue

                existing_thread["state"] = event["state"]
                existing_thread["text"] = event["text"]
                existing_thread["chapter"] = event["chapter"]
                existing_thread["tokens"] = event["tokens"]
                existing_thread["narrative_category"] = event["narrative_category"]
                merged_events.append(event)

    return merged_events



def render_timeline_overview(chapters: list[dict[str, Any]]) -> str:
    """Render a readable, chronological story map from canon memory."""
    timeline_events = merge_timeline_events(chapters)
    if not timeline_events:
        return "No timeline events recorded yet."

    lines: list[str] = []
    current_chapter: int | None = None
    for event in timeline_events:
        if event["chapter"] != current_chapter:
            if lines:
                lines.append("")
            current_chapter = event["chapter"]
            lines.append(f"Chapter {current_chapter}")

        bullet = "✓" if event["state"] == "RESOLVED" else "•"
        lines.append(
            f"{bullet} [{event['narrative_category']}] {event['text']} ({event['state']})"
        )

    return "\n".join(lines)


def handle_timeline_view() -> None:
    ensure_project_files()

    if not CANON_MEMORY_PATH.exists():
        print("No canon memory found.")
        return

    content = CANON_MEMORY_PATH.read_text(encoding="utf-8")
    chapters = parse_canon_memory(content)

    print("\nTIMELINE OVERVIEW\n")
    print(render_timeline_overview(chapters))


def handle_story_state() -> None:
    ensure_project_files()

    if not CANON_MEMORY_PATH.exists():
        print("No canon memory found.")
        return

    content = CANON_MEMORY_PATH.read_text(encoding="utf-8")
    chapters = parse_canon_memory(content)

    print("\nSTORY STATE\n")

    active_found = False

    for chapter in sorted(chapters, key=lambda c: c["number"]):
        for story_state in chapter.get("story_states", []):
            if str(story_state.get("state", "ACTIVE")).upper() != "ACTIVE":
                continue
            description = str(story_state.get("description", "")).strip()
            if not description:
                continue
            active_found = True
            print(f"- {description}")
            print(f"  FIRST_SEEN: Chapter {int(story_state.get('first_seen', chapter['number']))}")
            print()

    if not active_found:
        print("No active narrative states recorded.")


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


def handle_export_book_docx() -> None:
    """Export all numbered chapter files to a DOCX manuscript."""
    ensure_project_files()

    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.shared import Inches, Pt
    except ImportError:
        print("python-docx not installed. Run: pip install python-docx")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found in ~/writing/novel_project/chapters/")
        return
    warn_for_missing_chapter_files(chapter_paths)

    title = "Untitled Novel"
    if CANON_MEMORY_PATH.exists() and CANON_MEMORY_PATH.is_file():
        canon_lines = CANON_MEMORY_PATH.read_text(encoding="utf-8").splitlines()
        if canon_lines:
            raw_first_line = clean_terminal_text(canon_lines[0])
            if raw_first_line:
                if raw_first_line.lower().startswith("title:"):
                    parsed_title = raw_first_line.partition(":")[2].strip()
                    if parsed_title:
                        title = parsed_title
                else:
                    title = raw_first_line

    document = Document()

    normal_style = document.styles["Normal"]
    normal_font = normal_style.font
    normal_font.name = "Times New Roman"
    normal_font.size = Pt(12)
    normal_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    normal_style.paragraph_format.first_line_indent = Inches(0.5)

    for section in document.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    title_paragraph = document.add_paragraph()
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_paragraph.add_run(title)

    subtitle_paragraph = document.add_paragraph()
    subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_paragraph.add_run("A Novel")

    by_paragraph = document.add_paragraph()
    by_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    by_paragraph.add_run("by")

    blank_author_paragraph = document.add_paragraph()
    blank_author_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    blank_author_paragraph.add_run("")

    document.add_page_break()

    scene_break_pattern = re.compile(r"^(?:-{3,}|\*{3,}|(?:\*\s*){3,}|(?:—\s*){3,})$")

    for path in chapter_paths:
        chapter_number = extract_chapter_number(path)
        if chapter_number is None:
            continue

        chapter_text = clean_terminal_text(path.read_text(encoding="utf-8"))

        document.add_page_break()
        heading = document.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.paragraph_format.first_line_indent = Inches(0)
        heading_run = heading.add_run(f"CHAPTER {chapter_number}")
        heading_run.font.name = "Times New Roman"
        heading_run.font.size = Pt(16)
        heading_run.bold = True

        for raw_line in chapter_text.splitlines():
            line = raw_line.strip()
            if not line:
                document.add_paragraph("")
                continue

            if scene_break_pattern.fullmatch(line):
                scene_break = document.add_paragraph()
                scene_break.alignment = WD_ALIGN_PARAGRAPH.CENTER
                scene_break.paragraph_format.first_line_indent = Inches(0)
                scene_break.add_run("---")
                continue

            body_paragraph = document.add_paragraph(line)
            body_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

    output_path = MANUSCRIPT_DIR / "novel.docx"
    if output_path.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = MANUSCRIPT_DIR / f"novel_{timestamp}.docx"

    try:
        document.save(str(output_path))
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Export failed: {exc}")
        return

    output_display = str(output_path).replace(str(Path.home()), "~", 1)
    print("DOCX manuscript export complete:")
    print(output_display)


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
    print()
    print("WRITING")
    print("/scene-summary")
    print("/proofread")
    print()
    print("MEMORY")
    print("/rebuild-memory")
    print("/story-state")
    print("/timeline-view")
    print()
    print("IDEAS")
    print("/ideas")
    print("/idea-resurface")
    print()
    print("DRAFTS")
    print("/draft-save")
    print("/draft-list")
    print("/draft-load")
    print("/draft-pass --structure")
    print("/draft-pass --tension")
    print()
    print("BOOK")
    print("/build-book")
    print("/export-book --docx")
    print("/book-integrity")
    print()
    print("SYSTEM")
    print("/novel-stats")
    print("/help")
    print("exit")


def handle_export_book(command_text: str = "") -> None:
    """Route /export-book options to concrete export handlers."""
    if "--docx" in command_text:
        handle_export_book_docx()
        return
    print("Unsupported export option. Use /export-book --docx")



def main() -> None:
    """Run the terminal assistant."""
    ensure_project_files()

    try:
        client = create_client()
    except RuntimeError as exc:
        print(exc)
        return

    conversation_history: list[dict[str, str]] = []

    command_handlers: dict[str, Callable[[str], None]] = {
        "/scene-summary": lambda command_text="": handle_scene_summary(client),
        "/chapter-summary": lambda command_text="": handle_chapter_summary(client),
        "/rebuild-summaries": lambda command_text="": handle_rebuild_summaries(client),
        "/rebuild-memory": lambda command_text="": handle_rebuild_memory(client, command_text),
        "/continuity-check": lambda command_text="": handle_continuity_check(client),
        "/book-integrity": lambda command_text="": handle_book_integrity(client),
        "/proofread": lambda command_text="": handle_proofread(client),
        "/idea-resurface": lambda command_text="": handle_idea_resurface(client),
        "/draft-pass": lambda command_text="": handle_draft_pass(client, command_text),
        "/build-book": lambda command_text="": handle_build_book(),
        "/draft-save": lambda command_text="": handle_draft_save(),
        "/draft-list": lambda command_text="": handle_draft_list(),
        "/draft-load": lambda command_text="": handle_draft_load(),
        "/save-draft": lambda command_text="": handle_draft_save(),
        "/drafts": lambda command_text="": handle_draft_list(),
        "/restore-draft": lambda command_text="": handle_draft_load(),
        "/ideas": lambda command_text="": handle_ideas(),
        "/world-add": lambda command_text="": handle_world_add(),
        "/timeline-view": lambda command_text="": handle_timeline_view(),
        "/story-state": lambda command_text="": handle_story_state(),
        "/export-chapter": lambda command_text="": handle_export_chapter(),
        "/export-book": handle_export_book,
        "/novel-stats": lambda command_text="": handle_novel_stats(),
        "/help": lambda command_text="": print_help(),
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


        routed = False
        for command_name in sorted(command_handlers.keys(), key=len, reverse=True):
            if user_input == command_name or user_input.startswith(f"{command_name} "):
                command_handlers[command_name](user_input)
                routed = True
                break

        if routed:
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
