import argparse
import os
from typing import List, Tuple

from dotenv import load_dotenv

from src.db.db_connection import db_connection
from src.topic.act_topic_handler import ActTopicHandler


load_dotenv()


def prompt_confirm(message: str) -> bool:
    """Prompt the user for a yes/no confirmation and return True for yes."""
    while True:
        response = input(f"{message} [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"", "n", "no"}:
            return False
        print("Please respond with 'y' or 'n'.")


def format_topics(topics: List[Tuple[str, str, int]]) -> str:
    lines = []
    for committee, topic, importance in topics:
        lines.append(f"  - {committee} :: {topic} (importance {importance})")
    return "\n".join(lines)


def delete_topics_for_act(act_title: str) -> None:
    handler = ActTopicHandler()
    topics = handler.fetch_topics_for_act(act_title)
    if not topics:
        print(f"No topics found for '{act_title}'. Nothing to delete.")
        return

    print(f"Found {len(topics)} topic(s) for '{act_title}':")
    print(format_topics(topics))

    if not prompt_confirm("Delete all topics listed above?"):
        print("Aborted. No changes made.")
        return

    deleted = handler.delete_topics_for_act(act_title)
    if deleted:
        print(f"Removed {deleted} topic(s) for '{act_title}'.")
    else:
        print(f"No topics were removed for '{act_title}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all topics for a given Act title.")
    parser.add_argument("--title", help="Exact Act title as stored in act_topics/legislations")
    args = parser.parse_args()

    act_title = args.title or os.getenv("TOPIC_ACT") or os.getenv("CORE_ACT")
    if not act_title:
        act_title = input("Enter the exact Act title: ").strip()

    if not act_title:
        print("Act title is required. Exiting.")
        return

    try:
        delete_topics_for_act(act_title.strip())
    finally:
        db_connection.close_all_connections()
        print("Database connections closed.")


if __name__ == "__main__":
    main()
