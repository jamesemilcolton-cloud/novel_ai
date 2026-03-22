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
ANSI_ESCAPE_PATTERN = re.compile(r"\033\[[0-9;]*[A-Za-z]")
BRACKETED_PASTE_PATTERN = re.compile(r"(?:\033\[|\^\[\[?)(?:200~|201~|E)|\[\[200~|\[\[201~")
DISALLOWED_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAIN_TEMPERATURE = 0.8
SCENE_TEMPERATURE = 0.1
CONTINUITY_TEMPERATURE = 0.0
PROOFREAD_TEMPERATURE = 0.1
IDEA_SUGGEST_TEMPERATURE = 0.4

MAIN_SYSTEM_PROMPT = """You are a thoughtful AI novel-writing assistant.
Help the user think through story ideas, scenes, structure, tone, character, and prose.
Use the provided memory carefully and naturally.
Be creative, clear, and practical.
Do not invent persistent facts unless the user states them.
"""

SCENE_SYSTEM_PROMPT = """You are a strict isolated extraction tool for a novel-writing project.

Your job is to read ONLY the pasted scene text from this one request and extract possible canon memory facts.
Do not give writing advice.
Do not critique the scene.
Do not rewrite the scene.
Do not summarize previous outputs.
Do not refer to any chat history, persona, or prior memory.
Use ONLY the current scene provided by the user in this one request.

Return output in this exact structure:

Memory suggestions:

1. fact text -> [Category]
2. fact text -> [Category]
3. fact text -> [Category]

Rules:
- Use short, concrete facts only.
- Every suggestion must include exactly one category tag in square brackets.
- Allowed categories include Context, World, Location, Timeline, Character, Relationship, Object, or other concise canon-memory labels when needed.
- If there are no strong canon facts, return exactly:
Memory suggestions:

None
- Return ONLY the Memory suggestions output.
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

IDEA_SUGGEST_SYSTEM_PROMPT = """You are an assistant helping a novelist integrate ideas.

Your job:
- Read the current chapter
- Read the list of ideas
- Suggest ONLY ideas that naturally fit the current story direction

Rules:
- Do NOT force ideas
- Do NOT invent connections
- Do NOT suggest irrelevant ideas
- If no ideas fit, say exactly:

No ideas that could work here.

Return format:

IDEA SUGGESTIONS

1. idea → short reason
2. idea → short reason
"""


# ============================================================
# Filesystem helpers
# ============================================================


def ensure_project_files() -> None:
    """Create the expected project folders and files if they do not already exist."""
    PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    CONTINUITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REBUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CANON_MEMORY_PATH.touch(exist_ok=True)
    SCENE_SUMMARIES_PATH.touch(exist_ok=True)



def read_text_file(path: Path) -> str:
    """Read a text file and return cleaned text."""
    text = path.read_text(encoding="utf-8").strip()
    return text if text else "(empty)"



def load_memory_block() -> str:
    """Load canonical story memory for the main assistant and continuity checker."""
    return read_text_file(CANON_MEMORY_PATH)


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
                current_chapter["categories"][current_category].append(fact)

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
                lines.append(f"- {fact}")
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks).rstrip() + ("\n" if blocks else "")



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

    for fact, category in cleaned_facts:
        target_chapter["categories"].setdefault(category, []).append(fact)

    CANON_MEMORY_PATH.write_text(render_canon_memory(chapters), encoding="utf-8")
    print(f"Saved {len(cleaned_facts)} canon fact(s) to {CANON_MEMORY_PATH}.")



def append_scene_summary(chapter_number: int, summary_text: str) -> None:
    """Append scene extraction output to the non-canonical storage log."""
    ensure_project_files()
    cleaned_summary = summary_text.strip()
    if not cleaned_summary:
        return

    with SCENE_SUMMARIES_PATH.open("a", encoding="utf-8") as file:
        if SCENE_SUMMARIES_PATH.stat().st_size > 0:
            file.write("\n\n" + ("-" * 40) + "\n\n")
        file.write(f"CHAPTER {chapter_number}\n\n{cleaned_summary}\n")



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
    categories: OrderedDict[str, list[str]] = OrderedDict()
    for _, fact, category in suggestions:
        cleaned_fact = fact.strip()
        cleaned_category = category.strip()
        if not cleaned_fact or not cleaned_category:
            continue
        categories.setdefault(cleaned_category, []).append(cleaned_fact)

    return {
        "number": chapter_number,
        "categories": categories,
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
    log_path = REBUILD_LOG_DIR / f"rebuild_{timestamp.strftime('%Y%m%d_%H%M')}.txt"
    content = "\n".join(
        [
            f"mode: {mode}",
            *lines,
            f"timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
    )
    log_path.write_text(content + "\n", encoding="utf-8")
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



def prompt_for_chapter_number() -> int | None:
    """Ask the user for a chapter number."""
    print("Chapter number?")
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



def parse_memory_suggestions(result: str) -> list[tuple[int, str, str]]:
    """Parse numbered memory suggestions from the scene extractor output."""
    suggestions: list[tuple[int, str, str]] = []
    for match in SUGGESTION_PATTERN.finditer(result):
        number = int(match.group(1))
        fact = match.group(2).strip()
        category = match.group(3).strip()
        suggestions.append((number, fact, category))
    return suggestions



def prompt_for_selection(max_number: int) -> list[int] | None:
    """Ask the user which extracted facts should be saved."""
    print("\nSelect numbers to save:")
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



def build_scene_messages(scene_text: str) -> list[dict[str, str]]:
    """Build the isolated message list for scene analysis."""
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


def build_idea_suggest_messages(
    chapter_text: str,
    ideas_block: str,
    canon_memory_block: str,
) -> list[dict[str, str]]:
    """Build the message list for suggesting relevant saved ideas."""
    return [
        {
            "role": "system",
            "content": IDEA_SUGGEST_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Current chapter text:\n\n{chapter_text}\n\n"
                f"Ideas list:\n\n{ideas_block}\n\n"
                f"Canon memory:\n\n{canon_memory_block}"
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



# ============================================================
# Command handlers
# ============================================================


def handle_scene_summary(client: Any) -> None:
    """Analyse one pasted scene in a fully isolated request."""
    chapter_number = prompt_for_chapter_number()
    if chapter_number is None:
        return

    print("Paste scene. Type END when finished.")
    scene_text = collect_multiline_input(end_marker="END")
    scene_text = clean_terminal_text(scene_text)

    if not scene_text:
        print("No scene entered.")
        return

    try:
        result = request_chat_completion(
            client=client,
            messages=build_scene_messages(scene_text),
            temperature=SCENE_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Scene summary failed: {exc}")
        return

    try:
        append_scene_summary(chapter_number, result)
    except OSError as exc:
        print(f"Scene summary generated, but could not save log: {exc}")
        print()
        print(result)
        return

    print()
    print(result)

    suggestions = parse_memory_suggestions(result)
    if not suggestions:
        print(f"\nScene summary log saved to {SCENE_SUMMARIES_PATH}.")
        print("No structured canon suggestions found. Nothing saved.")
        return

    selection_numbers = prompt_for_selection(max(number for number, _, _ in suggestions))
    if selection_numbers is None:
        print(f"Scene summary log saved to {SCENE_SUMMARIES_PATH}.")
        print("Nothing saved.")
        return

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

    print(f"Scene summary log saved to {SCENE_SUMMARIES_PATH}.")



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
    ]

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
        report_path.write_text(report + "\n", encoding="utf-8")
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
            CANON_MEMORY_PATH.write_text(
                render_canon_memory(rebuilt_chapters),
                encoding="utf-8",
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
            CANON_MEMORY_PATH.write_text(render_canon_memory(chapters), encoding="utf-8")
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


def handle_idea_suggest(client: Any) -> None:
    """Suggest saved ideas that fit the selected chapter."""
    ensure_project_files()
    print("Enter chapter number:")
    try:
        raw_value = input("> ").strip()
    except EOFError:
        print()
        return

    if not raw_value:
        print("No chapter number entered.")
        return

    if not raw_value.isdigit():
        print("Chapter number must be a positive integer.")
        return

    chapter_number = int(raw_value)
    if chapter_number <= 0:
        print("Chapter number must be a positive integer.")
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
    messages = build_idea_suggest_messages(
        chapter_text=chapter_text,
        ideas_block=ideas_block,
        canon_memory_block=canon_memory_block,
    )

    try:
        result = request_chat_completion(
            client=client,
            messages=messages,
            temperature=IDEA_SUGGEST_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Idea suggestion failed: {exc}")
        return

    print()
    print(result)



def handle_build_book() -> None:
    """Compile all numbered chapter files into a single manuscript file."""
    if not CHAPTERS_DIR.exists():
        print("No chapter files found.")
        return

    chapter_paths = load_sorted_chapter_paths()
    if not chapter_paths:
        print("No chapter files found.")
        return

    manuscript_text = build_manuscript_text(chapter_paths)

    try:
        MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        MANUSCRIPT_PATH.write_text(manuscript_text, encoding="utf-8")
    except OSError as exc:
        print(f"Manuscript build failed: {exc}")
        return

    print("Manuscript built successfully.")



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
    print("  /rebuild-memory")
    print("  /continuity-check")
    print("  /proofread")
    print("  /idea-suggest")
    print("  /build-book")
    print("  /ideas")
    print("  /world-add")
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
        "/rebuild-memory": lambda: handle_rebuild_memory(client),
        "/continuity-check": lambda: handle_continuity_check(client),
        "/proofread": lambda: handle_proofread(client),
        "/idea-suggest": lambda: handle_idea_suggest(client),
        "/build-book": handle_build_book,
        "/ideas": handle_ideas,
        "/world-add": handle_world_add,
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

        if user_input in command_handlers:
            command_handlers[user_input]()
            continue

        memory_block = load_memory_block()
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
