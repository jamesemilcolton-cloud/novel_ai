from __future__ import annotations

import os
import re
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
CONTINUITY_REPORTS_DIR = PROJECT_ANALYSIS_DIR / "continuity_reports"
CHAPTER_FILENAME_PATTERN = re.compile(r"chapter_(\d+)\.txt$")

MEMORY_FILES = {
    "context": PROJECT_MEMORY_DIR / "context.txt",
    "world": PROJECT_MEMORY_DIR / "world.txt",
    "ideas": PROJECT_MEMORY_DIR / "ideas.txt",
    "timeline": PROJECT_MEMORY_DIR / "timeline.txt",
}

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAIN_TEMPERATURE = 0.8
SCENE_TEMPERATURE = 0.1
CONTINUITY_TEMPERATURE = 0.0

MAIN_SYSTEM_PROMPT = """You are a thoughtful AI novel-writing assistant.
Help the user think through story ideas, scenes, structure, tone, character, and prose.
Use the provided memory carefully and naturally.
Be creative, clear, and practical.
Do not invent persistent facts unless the user states them.
"""

SCENE_SYSTEM_PROMPT = """You are a strict isolated analysis tool for a novel-writing project.

Your job is to read ONLY the pasted scene text from this one request and extract possible long-term memory items.
Do not give writing advice.
Do not critique the scene.
Do not rewrite the scene.
Do not summarize previous outputs.
Do not refer to any chat history, persona, or prior memory.
Use ONLY the current scene provided by the user in this one request.

Return output in this exact structure:

### MEMORY SUGGESTIONS

Context:
- ...

Timeline:
- ...

World:
- ...

Ideas:
- ...

Rules:
- Include only concrete memory candidates or clearly labeled story ideas.
- Keep each bullet short and specific.
- If a section has nothing useful, write:
- None
- Return ONLY the structured MEMORY SUGGESTIONS output.
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


# ============================================================
# Filesystem helpers
# ============================================================


def ensure_project_files() -> None:
    """Create the expected project folders and files if they do not already exist."""
    PROJECT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    CONTINUITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for path in MEMORY_FILES.values():
        path.touch(exist_ok=True)




def read_text_file(path: Path) -> str:
    """Read a text file and return cleaned text."""
    text = path.read_text(encoding="utf-8").strip()
    return text if text else "(empty)"



def load_memory_block() -> str:
    """Load canonical continuity memory for the main assistant."""
    sections = []
    for name, path in MEMORY_FILES.items():
        title = name.capitalize()
        sections.append(f"{title}:\n{read_text_file(path)}")
    return "\n\n".join(sections)



def append_memory_fact(memory_key: str, fact: str) -> None:
    """Append one fact to the chosen canonical memory file."""
    ensure_project_files()
    cleaned_fact = fact.strip()
    if not cleaned_fact:
        print("Nothing saved.")
        return

    path = MEMORY_FILES[memory_key]
    with path.open("a", encoding="utf-8") as file:
        file.write(cleaned_fact + "\n")

    print(f"Saved to {path}.")



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


# ============================================================
# Input helpers
# ============================================================


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

    return "\n".join(lines).strip()



def prompt_for_fact() -> str:
    """Ask the user for a fact to save."""
    print("Enter fact to save:")
    try:
        return input("> ")
    except EOFError:
        print()
        return ""


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
                f"Previous chapters:\n\n{previous_chapters_block}\n\n"
                f"Selected chapter ({selected_chapter_name}):\n\n{selected_chapter_text}"
            ),
        },
    ]


# ============================================================
# Command handlers
# ============================================================


def handle_save_command(memory_key: str) -> None:
    """Save one fact into a memory file."""
    fact = prompt_for_fact()
    append_memory_fact(memory_key, fact)



def handle_scene_summary(client: Any) -> None:
    """Analyse one pasted scene in a fully isolated request."""
    print("Enter scene to analyse. Type END on a new line when finished:")
    scene_text = collect_multiline_input(end_marker="END")

    if not scene_text:
        print("No scene entered.")
        return

    messages = build_scene_messages(scene_text)

    try:
        result = request_chat_completion(
            client=client,
            messages=messages,
            temperature=SCENE_TEMPERATURE,
        )
    except Exception as exc:  # Keep terminal app stable for the user.
        print(f"Scene summary failed: {exc}")
        return

    print()
    print(result)



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
    previous_chapters_block = format_chapter_block(previous_paths)
    selected_chapter_text = selected_path.read_text(encoding="utf-8").strip()
    messages = build_continuity_messages(
        memory_block=memory_block,
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

    print(f"Continuity report saved to {report_path}.")


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
    print("  /continuity-check")
    print("  /save-context")
    print("  /save-world")
    print("  /save-idea")
    print("  /save-timeline")
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
        "/continuity-check": lambda: handle_continuity_check(client),
        "/save-context": lambda: handle_save_command("context"),
        "/save-world": lambda: handle_save_command("world"),
        "/save-idea": lambda: handle_save_command("ideas"),
        "/save-timeline": lambda: handle_save_command("timeline"),
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
