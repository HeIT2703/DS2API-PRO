"""Small DS2API demo for real chat.deepseek.com sessions.

Set DEEPSEEK_USER_TOKEN before running this file. The token is never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DS2API import DeepSeekClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small DS2API demo against chat.deepseek.com.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Reply with one short sentence: DS2API is ready.",
        help="Prompt to send. Defaults to a short readiness check.",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Use expert mode instead of instant mode.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepThink for the request.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable web search for the request.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response text as it arrives.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Upload and attach a local file to the prompt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("DEEPSEEK_USER_TOKEN")
    if not token:
        print("Set DEEPSEEK_USER_TOKEN before running this demo.", file=sys.stderr)
        return 2

    model = "expert" if args.expert else "instant"
    ref_file_ids: list[str] = []

    with DeepSeekClient(token=token) as client:
        if args.file:
            upload = client.file.upload_file(str(args.file), wait_ready=True)
            file_info = client.file.extract_file_info(upload)
            ref_file_ids.append(file_info.id)
            print(f"Attached file: {file_info.file_name or args.file.name} ({file_info.status})")

        if args.stream:
            for event in client.ask_stream(
                args.prompt,
                model=model,
                thinking=args.thinking,
                search=args.search,
                ref_file_ids=ref_file_ids,
            ):
                if event.event_type == "RESPONSE_TEXT" and event.content:
                    print(event.content, end="", flush=True)
            print()
            return 0

        response = client.ask(
            args.prompt,
            model=model,
            thinking=args.thinking,
            search=args.search,
            ref_file_ids=ref_file_ids,
        )
        print(response.text)
        if response.citations:
            print("\nCitations:")
            for citation in response.citations:
                print(f"- {citation.title}: {citation.url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
