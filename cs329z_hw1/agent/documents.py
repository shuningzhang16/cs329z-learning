"""Passage search and bounded, exact-line access to company documents."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import CHARS_PER_TOKEN


SNIPPET_CHARS = 150 * CHARS_PER_TOKEN
WINDOW_CHARS = 1500 * CHARS_PER_TOKEN
_STOPWORDS = set("a an the is are was were be been being am do does did for in on at to of "
                "and or as by with from this that it its my i we our me you your how what "
                "when where which who why can may will would should have has had".split())


def build_doc_index(docs: list[dict]) -> dict[str, Any]:
    """Index paragraphs with their document and section titles as context."""
    from cs329z_hw1.adapters import run_bm25_build

    documents = {}
    passages = []
    texts = []
    frequencies: Counter[str] = Counter()
    for doc in docs:
        lines = doc["text"].splitlines()
        documents[doc["doc_id"]] = {**doc, "lines": lines}
        frequencies.update(set(tokenize(doc["text"])))
        section = ""
        section_line = 1
        start = 0
        while start < len(lines):
            if not lines[start].strip():
                start += 1
                continue
            begin = start
            while start < len(lines) and (not lines[start].strip() or lines[start].startswith("#")):
                if lines[start].startswith("#"):
                    section = lines[start].lstrip("# ").strip()
                    section_line = start + 1
                start += 1
            end = min(start + 1, len(lines))
            while end < len(lines) and lines[end].strip() and not lines[end].startswith("#"):
                end += 1
            text = "\n".join(lines[begin:end])
            passages.append({
                "doc_id": doc["doc_id"], "title": doc["title"],
                "start_line": begin + 1, "end_line": end, "text": text,
                "section_line": section_line,
                "terms": set(tokenize(text)),
            })
            texts.append(f"{doc['title']}\n{section}\n{text}")
            start = end

    return {
        "documents": documents,
        "passages": passages,
        "bm25": run_bm25_build(texts),
        "idf": {term: math.log(1 + len(docs) / count) for term, count in frequencies.items()},
    }


def _snippet(text: str, terms: set[str], idf: dict[str, float]) -> str:
    """Choose a contiguous excerpt around the best concentration of query terms."""
    if len(text) <= SNIPPET_CHARS:
        return text
    matches = [(match.start(), match.end(), match.group())
               for match in re.finditer(r"[a-z0-9]+", text.lower())
               if match.group() in terms]
    starts = {0}
    for position, _, _ in matches:
        starts.add(max(0, min(position - 100, len(text) - SNIPPET_CHARS)))
    best_start = max(sorted(starts), key=lambda start: sum(
        idf.get(term, 1.0)
        for term in {term for left, right, term in matches
                     if left >= start and right <= start + SNIPPET_CHARS}
    ))
    # Keep words whole when a window begins or ends within a word.
    end = min(best_start + SNIPPET_CHARS, len(text))
    while best_start > 0 and best_start < end and text[best_start - 1].isalnum() and text[best_start].isalnum():
        best_start += 1
    while end < len(text) and end > best_start and text[end - 1].isalnum() and text[end].isalnum():
        end -= 1
    return text[best_start:end]


def search_docs(index: dict[str, Any], query: str, k: int = 5) -> list[dict]:
    """Return one relevant excerpt per document, with line pointers for reading."""
    from cs329z_hw1.adapters import run_bm25_search

    if k <= 0:
        return []
    terms = set(tokenize(query))
    if not terms:
        return []
    terms = terms - _STOPWORDS or terms
    ranked = run_bm25_search(index["bm25"], " ".join(sorted(terms)), len(index["passages"]))
    results = []
    seen = set()
    for passage_id, _score in ranked:
        passage = index["passages"][passage_id]
        if passage["doc_id"] in seen or not terms.intersection(passage["terms"]):
            continue
        snippet = _snippet(passage["text"], terms, index["idf"])
        results.append({
            "doc_id": passage["doc_id"],
            "title": passage["title"],
            "snippet": snippet,
            "start_line": passage["section_line"],
            "end_line": passage["end_line"],
        })
        seen.add(passage["doc_id"])
        if len(results) == k:
            break
    return results


def read_doc(index: dict[str, Any], doc_id: str, start_line: int = 1) -> dict:
    """Read complete lines, stopping before the 1,500-token window limit."""
    lines = index["documents"][doc_id]["lines"]
    if not isinstance(start_line, int) or not 1 <= start_line <= len(lines):
        raise ValueError(f"start_line must be between 1 and {len(lines)}")
    end = start_line - 1
    chars = 0
    while end < len(lines):
        added = len(lines[end]) + (1 if end >= start_line else 0)
        if chars + added > WINDOW_CHARS:
            break
        chars += added
        end += 1
    if end < start_line:
        raise ValueError(f"Line {start_line} exceeds the document window limit")
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end,
        "total_lines": len(lines),
        "text": "\n".join(lines[start_line - 1:end]),
    }
