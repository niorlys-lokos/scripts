#!/usr/bin/env python3
"""
Inspect an SFT parquet dataset and print concrete examples of common quality issues.

Usage:
    python inspect_sft_issues.py /path/to/file.parquet

Example:
    python inspect_sft_issues.py sft-output-20260306-cohortbcd_sft_dataset_shard-000000.parquet
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


MAX_EXAMPLES_PER_SECTION = 5
LONG_SYSTEM_THRESHOLD = 4000  # chars
PREVIEW_LEN = 800


def preview(text: Any, n: int = PREVIEW_LEN) -> str:
    s = str(text)
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")


def load_messages(value: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Returns (messages, issue)
    issue is a human-readable explanation if parsing fails or format is unexpected.
    """
    if isinstance(value, list):
        return value, None

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed, None
            return None, f"JSON parsed but top-level type is {type(parsed).__name__}, expected list"
        except Exception as e:
            return None, f"messages is string but not valid JSON: {e}"

    return None, f"messages has unexpected type: {type(value).__name__}"


def get_first_message_by_role(messages: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    for m in messages:
        if m.get("role") == role:
            return m
    return None


def role_sequence(messages: List[Dict[str, Any]]) -> List[str]:
    return [str(m.get("role")) for m in messages]


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_example_header(row_idx: int, row: pd.Series) -> None:
    print(f"\n[row={row_idx}] id={row.get('id')} task_id={row.get('task_id')} sample_index={row.get('sample_index')}")


def detect_structured_data_in_text(text: str) -> bool:
    """
    Heuristic: finds lines that look like 'key: value'
    especially snake_case keys or field-like metadata.
    """
    field_like_lines = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_ ]{1,40}:\s+.+$", line):
            field_like_lines += 1
        if field_like_lines >= 3:
            return True
    return False


def detect_asr_artifacts(text: str) -> List[str]:
    """
    Returns a list of matched artifact patterns.
    """
    patterns = []

    # repeated short phrase: "Yeah. Yeah." / "I don't. I don't"
    if re.search(r"\b([A-Za-z']+(?:\s+[A-Za-z']+){0,3})\.\s+\1\b", text, flags=re.IGNORECASE):
        patterns.append("repeated_phrase")

    # filler words
    if re.search(r"\b(um|uh|hmm|mm-hmm|you know|like)\b", text, flags=re.IGNORECASE):
        patterns.append("filler_words")

    # false starts with period break: "I'll be. I'll be curious"
    if re.search(r"\b([A-Za-z']+(?:\s+[A-Za-z']+){0,3})\.\s+\1\s+", text, flags=re.IGNORECASE):
        patterns.append("false_start_repeat")

    # doubled acknowledgement pattern
    if re.search(r"\b(yeah|okay|ok|right|sure)\.\s+\1\b", text, flags=re.IGNORECASE):
        patterns.append("doubled_ack")

    return patterns


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python inspect_sft_issues.py /path/to/file.parquet")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    print_section("DATASET SUMMARY")
    print(f"File: {parquet_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes)

    parsed_messages: Dict[int, Optional[List[Dict[str, Any]]]] = {}
    parse_issues: List[Tuple[int, str]] = []

    for idx, row in df.iterrows():
        msgs, issue = load_messages(row.get("messages"))
        parsed_messages[idx] = msgs
        if issue:
            parse_issues.append((idx, issue))

    # 1. messages stored as JSON strings / parsing issues
    print_section("ISSUE 1: messages stored as JSON strings or malformed")
    found_string_examples = 0
    for idx, row in df.iterrows():
        raw = row.get("messages")
        if isinstance(raw, str):
            print_example_header(idx, row)
            print("messages Python type: str")
            print("Raw preview:")
            print(preview(raw))
            found_string_examples += 1
            if found_string_examples >= MAX_EXAMPLES_PER_SECTION:
                break

    if found_string_examples == 0:
        print("No string examples found.")

    if parse_issues:
        print("\nAdditional parse issues:")
        for idx, issue in parse_issues[:MAX_EXAMPLES_PER_SECTION]:
            row = df.iloc[idx]
            print_example_header(idx, row)
            print(issue)
            print("Raw messages preview:")
            print(preview(row.get("messages")))

    # 2. oversized system prompts
    print_section("ISSUE 2: oversized system prompts")
    long_system_count = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue
        sys_msg = get_first_message_by_role(msgs, "system")
        if not sys_msg:
            continue
        content = str(sys_msg.get("content", ""))
        if len(content) >= LONG_SYSTEM_THRESHOLD:
            print_example_header(idx, row)
            print(f"System prompt length: {len(content)} chars")
            print("System prompt preview:")
            print(preview(content))
            long_system_count += 1
            if long_system_count >= MAX_EXAMPLES_PER_SECTION:
                break

    if long_system_count == 0:
        print(f"No system prompts >= {LONG_SYSTEM_THRESHOLD} chars found.")

    # 3. structured data embedded in system prompt
    print_section("ISSUE 3: structured data embedded in system prompt text")
    structured_examples = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue
        sys_msg = get_first_message_by_role(msgs, "system")
        if not sys_msg:
            continue
        content = str(sys_msg.get("content", ""))
        if detect_structured_data_in_text(content):
            print_example_header(idx, row)
            print("Detected multiple field-like lines in system prompt.")
            print("System prompt preview:")
            print(preview(content))
            structured_examples += 1
            if structured_examples >= MAX_EXAMPLES_PER_SECTION:
                break

    if structured_examples == 0:
        print("No obvious structured-data-in-text examples found by heuristic.")

    # 4. assistant-first conversations
    print_section("ISSUE 4: assistant-first conversation ordering")
    assistant_first_count = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue

        non_system = [m for m in msgs if m.get("role") != "system"]
        if not non_system:
            continue

        first_role = non_system[0].get("role")
        if first_role == "assistant":
            print_example_header(idx, row)
            print("Role sequence:")
            print(role_sequence(msgs))
            print("First non-system messages:")
            for m in non_system[:4]:
                print(f"  - {m.get('role')}: {preview(m.get('content', ''), 180)}")
            assistant_first_count += 1
            if assistant_first_count >= MAX_EXAMPLES_PER_SECTION:
                break

    if assistant_first_count == 0:
        print("No assistant-first examples found.")

    # 5. ASR artifacts in user messages
    print_section("ISSUE 5: ASR / transcript artifacts in user messages")
    asr_count = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue

        matched = False
        for m in msgs:
            if m.get("role") != "user":
                continue
            content = str(m.get("content", ""))
            hits = detect_asr_artifacts(content)
            if hits:
                print_example_header(idx, row)
                print(f"Matched artifact types: {hits}")
                print("User message preview:")
                print(preview(content))
                matched = True
                asr_count += 1
                break

        if asr_count >= MAX_EXAMPLES_PER_SECTION:
            break

    if asr_count == 0:
        print("No ASR artifacts found by current heuristics.")

    # 6. very short assistant responses / fragmented target examples
    print_section("ISSUE 6: very short assistant responses")
    short_assistant_count = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue

        for m in msgs:
            if m.get("role") != "assistant":
                continue
            content = str(m.get("content", "")).strip()
            if 0 < len(content) <= 40:
                print_example_header(idx, row)
                print(f"Assistant response length: {len(content)}")
                print("Assistant response preview:")
                print(repr(content))
                short_assistant_count += 1
                break

        if short_assistant_count >= MAX_EXAMPLES_PER_SECTION:
            break

    if short_assistant_count == 0:
        print("No very short assistant messages found with current threshold.")

    # 7. duplicated metadata / redundant transcript_path
    print_section("ISSUE 7: duplicated metadata fields")
    dup_meta_count = 0
    for idx, row in df.iterrows():
        meta_raw = row.get("metadata_json")
        transcript_path = row.get("transcript_path")

        if not isinstance(meta_raw, str):
            continue

        try:
            meta = json.loads(meta_raw)
        except Exception:
            continue

        if isinstance(meta, dict) and "transcript_path" in meta and meta["transcript_path"] == transcript_path:
            print_example_header(idx, row)
            print(f"transcript_path column: {transcript_path}")
            print(f"metadata_json.transcript_path: {meta.get('transcript_path')}")
            print("metadata_json preview:")
            print(preview(meta_raw))
            dup_meta_count += 1
            if dup_meta_count >= MAX_EXAMPLES_PER_SECTION:
                break

    if dup_meta_count == 0:
        print("No duplicated transcript_path metadata found.")

    # 8. possible overlapping samples by task_id
    print_section("ISSUE 8: possible overlapping samples by task_id")
    grouped: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
    for idx, row in df.iterrows():
        grouped[str(row.get("task_id"))].append((idx, row.get("sample_index")))

    overlap_groups = []
    for task_id, items in grouped.items():
        if len(items) > 1:
            items_sorted = sorted(items, key=lambda x: (x[1] is None, x[1]))
            sample_indices = [x[1] for x in items_sorted if x[1] is not None]
            if len(sample_indices) >= 2:
                overlap_groups.append((task_id, items_sorted))

    if not overlap_groups:
        print("No repeated task_id groups found.")
    else:
        shown = 0
        for task_id, items in overlap_groups[:MAX_EXAMPLES_PER_SECTION]:
            print(f"\ntask_id={task_id}")
            print("Rows/sample_index values:")
            for idx, sample_index in items[:10]:
                row = df.iloc[idx]
                print(f"  row={idx} sample_index={sample_index} id={row.get('id')}")
            shown += 1
            if shown >= MAX_EXAMPLES_PER_SECTION:
                break

    # 9. consecutive duplicate roles
    print_section("ISSUE 9: consecutive duplicate roles")
    dup_role_count = 0
    for idx, row in df.iterrows():
        msgs = parsed_messages[idx]
        if not msgs:
            continue

        roles = role_sequence(msgs)
        bad_positions = []
        for i in range(1, len(roles)):
            if roles[i] == roles[i - 1] and roles[i] != "system":
                bad_positions.append(i)

        if bad_positions:
            print_example_header(idx, row)
            print("Role sequence:")
            print(roles)
            print("Duplicate role positions:", bad_positions)
            for pos in bad_positions[:3]:
                print(f"  Prev: {roles[pos-1]} -> {preview(msgs[pos-1].get('content', ''), 140)}")
                print(f"  Curr: {roles[pos]} -> {preview(msgs[pos].get('content', ''), 140)}")
            dup_role_count += 1
            if dup_role_count >= MAX_EXAMPLES_PER_SECTION:
                break

    if dup_role_count == 0:
        print("No consecutive duplicate non-system roles found.")

    print_section("DONE")
    print("This script prints examples, not full statistics.")
    print("If you want, the next step is to turn this into a validator that exports a CSV or markdown report.")


if __name__ == "__main__":
    main()