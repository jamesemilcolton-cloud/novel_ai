from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable


# ============================================================
# Configuration
# ============================================================

MEMORY_DIR = Path.home() / "writing"
MEMORY_FILES = {
    "context": MEMORY_DIR / "context.txt",
    "world": MEMORY_DIR / "world.txt",
    "ideas": MEMORY_DIR / "ideas.txt",
    "timeline": MEMORY_DIR / "timeline.txt",
}

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAIN_TEMPERATURE = 0.8
SCENE_TEMPERATURE = 0.2

MAIN_SYSTEM_PROMPT = """You are a thoughtful AI novel-writing assistant.
Help the user think through story ideas, scenes, structure, tone, character, and prose.
Use the provided memory carefully and naturally.
Be creative, clear, and practical.
Do not invent persistent facts unless the user states them.
"""

SCENE_SYSTEM_PROMPT = """You are a memory extraction tool for a novel-writing project.

Your job is to read ONLY the pasted scene and extract possible long-term memory items.
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
"""


# ============================================================
# Memory helpers
# ============================================================


def ensure_memory_files() -> None:
    """Create the memory folder and files if they do not already exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    for path in MEMORY_FILES.values():
        path.touch(exist_ok=True)



def read_memory_file(path: Path) -> str:
    """Read a memory file and return cleaned text."""
    text = path.read_text(encoding="utf-8").strip()
    return text if text else "(empty)"



def load_memory_block() -> str:
    """Combine all memory files into one block for the main chat assistant."""
    sections = []
    for name, path in MEMORY_FILES.items():
        title = name.capitalize()
        sections.append(f"{title}:\n{read_memory_file(path)}")
    return "\n\n".join(sections)



def append_memory_fact(memory_key: str, fact: str) -> None:
    """Append one fact to the chosen memory file."""
    cleaned_fact = fact.strip()
    if not cleaned_fact:
        print("Nothing saved.")
        return

    path = MEMORY_FILES[memory_key]
    with path.open("a", encoding="utf-8") as file:
        file.write(cleaned_fact + "\n")

    print(f"Saved to {path}.")


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
    """Build the isolated message list for scene memory extraction."""
    return [
        {
            "role": "system",
            "content": SCENE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"Scene to analyse for memory suggestions only:\n\n{scene_text}",
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
    """Run scene extraction in a fully isolated request."""
    print('Paste scene. Type END on a new line when finished.')
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
        print(f"Scene extraction failed: {exc}")
        return

    print()
    print(result)


# ============================================================
# Main application loop
# ============================================================


def print_welcome() -> None:
    """Show a simple startup message."""
    print("Novel AI Assistant")
    print(f"Model: {MODEL_NAME}")
    print("Type /help for commands or type exit to quit.")



def print_help() -> None:
    """Show available commands."""
    print("Available commands:")
    print("  /scene-summary")
    print("  /save-context")
    print("  /save-world")
    print("  /save-idea")
    print("  /save-timeline")
    print("  /help")
    print("  exit")



def main() -> None:
    """Run the terminal assistant."""
    ensure_memory_files()

    try:
        client = create_client()
    except RuntimeError as exc:
        print(exc)
        return

    conversation_history: list[dict[str, str]] = []

    command_handlers: dict[str, Callable[[], None]] = {
        "/scene-summary": lambda: handle_scene_summary(client),
        "/save-context": lambda: handle_save_command("context"),
        "/save-world": lambda: handle_save_command("world"),
        "/save-idea": lambda: handle_save_command("ideas"),
        "/save-timeline": lambda: handle_save_command("timeline"),
        "/help": print_help,
    }

    print_welcome()

    while True:
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
