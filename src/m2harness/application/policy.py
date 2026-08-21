"""Explicit egress policy: model APIs and web_search are separate capabilities."""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlsplit


class EgressCapability(StrEnum):
    MODEL_PROVIDER = "model.provider.call"
    WEB_SEARCH = "web.search"


class NetworkPolicy:
    def __init__(self, *, model_hosts: set[str] | frozenset[str] = frozenset(), search_hosts: set[str] | frozenset[str] = frozenset()) -> None:
        self.model_hosts = frozenset(host.lower() for host in model_hosts)
        self.search_hosts = frozenset(host.lower() for host in search_hosts)

    def authorize(self, capability: EgressCapability, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise PermissionError("only HTTPS egress with a hostname is permitted")
        allowed = self.model_hosts if capability == EgressCapability.MODEL_PROVIDER else self.search_hosts
        if parsed.hostname.lower() not in allowed:
            raise PermissionError(f"egress host is not allowlisted for {capability.value}: {parsed.hostname}")

    def authorize_model(self, url: str) -> None:
        self.authorize(EgressCapability.MODEL_PROVIDER, url)

    def authorize_search(self, url: str) -> None:
        self.authorize(EgressCapability.WEB_SEARCH, url)
