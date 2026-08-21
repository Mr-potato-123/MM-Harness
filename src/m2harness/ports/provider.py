"""Model provider port; provider SDKs stay behind this boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from m2harness.domain.agent import ModelCapabilities, ModelRequest, ModelResponse


class ModelProvider(Protocol):
    name: str

    def capabilities(self, model: str) -> ModelCapabilities: ...
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelResponse]: ...
