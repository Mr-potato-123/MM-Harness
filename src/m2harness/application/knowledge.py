"""Local knowledge providers.

This module intentionally has no network dependency.  HMML is loaded lazily,
validated as data, flattened into method entries, and searched with a bounded
deterministic lexical scorer.  The provider is suitable for a production
baseline because every hit carries the source path and immutable digest; an
embedding/reranking implementation can implement the same protocol later.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from m2harness.domain.knowledge import (
    KnowledgeEntry,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeSearchResult,
    KnowledgeSourceKind,
)


class KnowledgeBasePort(Protocol):
    def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult: ...


class EmptyKnowledgeBase:
    """A safe no-op provider used when an optional local index is unavailable."""

    def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(query=query.query, source_count=0, hits=(), truncated=False)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|[0-9]+|[\u3400-\u9fff]")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token.lower() for token in _TOKEN_RE.findall(value)))


def _string(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


class HMMLKnowledgeBase:
    """Read-only HMML JSON index compatible with MM-Agent's export format."""

    def __init__(self, path: Path, *, max_entries: int = 20_000) -> None:
        self.path = path.resolve()
        self.max_entries = max_entries
        self._signature: tuple[int, int] | None = None
        self._index_digest: str | None = None
        self._entries: tuple[KnowledgeEntry, ...] = ()

    @property
    def available(self) -> bool:
        return self.path.is_file() and not self.path.is_symlink()

    def _load_if_changed(self) -> None:
        if not self.available:
            self._entries = ()
            self._signature = None
            self._index_digest = None
            return
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, list):
            raise ValueError("HMML root must be a JSON array")
        entries: list[KnowledgeEntry] = []

        def visit(node: Any, hierarchy: tuple[str, ...]) -> None:
            if len(entries) >= self.max_entries or not isinstance(node, dict):
                return
            method_class = _string(node.get("method_class", "")).strip()
            current_hierarchy = hierarchy + ((method_class,) if method_class else ())
            method = _string(node.get("method", "")).strip()
            description = _string(node.get("description", "")).strip()
            if method and description:
                raw_entry_id = "hmml:" + "/".join((*current_hierarchy, method))
                suffix = hashlib.sha256(raw_entry_id.encode("utf-8")).hexdigest()[:16]
                entry_id = raw_entry_id[:280] + ":" + suffix
                entries.append(KnowledgeEntry(
                    entry_id=entry_id[:300], method=method[:500], description=description[:20_000],
                    hierarchy=current_hierarchy[:12], source=str(self.path), source_digest=digest,
                    metadata={"source_kind": KnowledgeSourceKind.HMML.value},
                ))
            children = node.get("children", ())
            if isinstance(children, list):
                for child in children:
                    visit(child, current_hierarchy)

        for item in value:
            visit(item, ())
        self._entries = tuple(entries)
        self._signature = signature
        self._index_digest = digest

    def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        self._load_if_changed()
        query_terms = set(_tokens(query.query))
        if not query_terms:
            return KnowledgeSearchResult(query=query.query, source_count=len(self._entries), index_digest=self._index_digest)
        scored: list[KnowledgeHit] = []
        for entry in self._entries:
            if query.source_kinds and KnowledgeSourceKind(entry.metadata.get("source_kind", "hmml")) not in query.source_kinds:
                continue
            method_terms = set(_tokens(entry.method))
            hierarchy_terms = set(_tokens(" ".join(entry.hierarchy)))
            description_terms = set(_tokens(entry.description))
            method_matches = query_terms & method_terms
            hierarchy_matches = query_terms & hierarchy_terms
            description_matches = query_terms & description_terms
            matched = tuple(sorted(method_matches | hierarchy_matches | description_matches))
            if not matched:
                continue
            score = 6.0 * len(method_matches) + 2.5 * len(hierarchy_matches) + 1.0 * len(description_matches)
            if query.query.lower() in entry.method.lower():
                score += 8.0
            snippet = entry.description[:2_000]
            scored.append(KnowledgeHit(entry=entry, score=score, matched_terms=matched, snippet=snippet))
        scored.sort(key=lambda hit: (-hit.score, hit.entry.method.lower(), hit.entry.entry_id))
        return KnowledgeSearchResult(
            query=query.query, hits=tuple(scored[:query.top_k]), source_count=len(self._entries),
            index_digest=self._index_digest, truncated=len(scored) > query.top_k,
        )


def default_hmml_path(root: Path | None = None) -> Path | None:
    """Find the optional MM-Agent HMML export without making it a dependency."""
    base = (root or Path.cwd()).resolve()
    candidates = (
        Path(os.environ["M2HARNESS_HMML_PATH"]).resolve()
        if os.environ.get("M2HARNESS_HMML_PATH") else None,
        base / "knowledge" / "HMML.json",
        base / "ref_github" / "LLM-MM-Agent" / "MMAgent" / "HMML" / "HMML.json",
        # A sibling clone is the common local development layout:
        # <parent>/MM-Harness and <parent>/LLM-MM-Agent.
        base.parent / "LLM-MM-Agent" / "MMAgent" / "HMML" / "HMML.json",
        base.parent / "MM" / "_Harness" / "ref_github" / "LLM-MM-Agent" / "MMAgent" / "HMML" / "HMML.json",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None
