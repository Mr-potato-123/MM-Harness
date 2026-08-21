"""Bounded, report-first DeepResearch orchestration for Model Agent context.

The service mirrors the useful DeerFlow research phases while staying local by
default: broad survey, dimension-specific deep dives, diversity checks, and
synthesis into citation-bearing findings.  A web adapter may be injected only
when the caller explicitly authorizes the ``web.search`` capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from m2harness.application.knowledge import KnowledgeBasePort
from m2harness.domain.knowledge import KnowledgeQuery
from m2harness.domain.research import ResearchFinding, ResearchReport, ResearchSource, ResearchSourceKind


class ResearchAgentPort(Protocol):
    def research(self, query: str, *, max_facets: int = 5, top_k: int = 6) -> ResearchReport: ...


class AuthorizedWebResearchPort(Protocol):
    def search(self, query: str, *, limit: int) -> Sequence[ResearchSource]: ...


@dataclass(frozen=True)
class DeepResearchService:
    knowledge_base: KnowledgeBasePort
    web_search: AuthorizedWebResearchPort | Callable[[str, int], Sequence[ResearchSource]] | None = None
    allow_web: bool = False
    max_sources: int = 64
    max_excerpt_chars: int = 1_500

    @staticmethod
    def plan_facets(query: str, *, max_facets: int) -> tuple[str, ...]:
        """Create stable research dimensions before any source is consulted."""
        normalized = " ".join(query.split())
        candidates = (
            normalized,
            normalized + " problem formulation assumptions variables objective constraints",
            normalized + " mathematical modeling methods alternatives baseline",
            normalized + " validation sensitivity uncertainty reproducibility",
            normalized + " implementation data analysis visualization expected outputs",
        )
        return tuple(dict.fromkeys(item[:20_000] for item in candidates))[:max(1, min(max_facets, 8))]

    def research(self, query: str, *, max_facets: int = 5, top_k: int = 6) -> ResearchReport:
        facets = self.plan_facets(query, max_facets=max_facets)
        sources: list[ResearchSource] = []
        seen_ids: set[str] = set()
        gaps: list[str] = []
        for index, facet in enumerate(facets):
            result = self.knowledge_base.search(KnowledgeQuery(query=facet, top_k=top_k))
            for hit in result.hits:
                source_id = "local:" + hit.entry.entry_id
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                sources.append(ResearchSource(
                    source_id=source_id, title=hit.entry.method, uri=hit.entry.source,
                    kind=ResearchSourceKind.LOCAL_KNOWLEDGE, trust="trusted-local-index",
                    excerpt=hit.snippet[:self.max_excerpt_chars], digest=hit.entry.source_digest,
                    metadata={"score": hit.score, "matched_terms": list(hit.matched_terms), "facet_index": index, "hierarchy": list(hit.entry.hierarchy)},
                ))
            if not result.hits:
                gaps.append("No local knowledge hit for facet: " + facet[:300])
        if self.allow_web and self.web_search is not None:
            for facet in facets:
                remote = self._web(facet, top_k)
                for source in remote:
                    if source.source_id not in seen_ids:
                        seen_ids.add(source.source_id)
                        sources.append(source)
        findings: list[ResearchFinding] = []
        # One finding per source keeps provenance explicit and avoids inventing
        # cross-source claims before a real Model Agent performs synthesis.
        bounded_sources = sources[:max(1, min(self.max_sources, 100))]
        for index, source in enumerate(bounded_sources):
            findings.append(ResearchFinding(
                finding_id=f"finding-{index + 1}",
                claim=f"Candidate method/evidence: {source.title}.",
                evidence_source_ids=(source.source_id,),
                rationale=source.excerpt[:2_000],
                confidence=0.65 if source.kind == ResearchSourceKind.LOCAL_KNOWLEDGE else 0.45,
            ))
        return ResearchReport(
            query=query, plan=facets, sources=tuple(bounded_sources), findings=tuple(findings),
            gaps=tuple(gaps[:20]), local_only=not (self.allow_web and self.web_search is not None),
            completed=True, index_digest=next((source.digest for source in sources if source.digest), None),
        )

    def _web(self, query: str, limit: int) -> Sequence[ResearchSource]:
        if self.web_search is None:
            return ()
        if hasattr(self.web_search, "search"):
            return self.web_search.search(query, limit=limit)  # type: ignore[union-attr]
        return self.web_search(query, limit)
