from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence

import httpx2


class ShardedTransport(httpx2.AsyncBaseTransport):
    """Connection pools taking requests in turn.

    `httpcore` scans all of a pool's connections per request, on the event loop; `n` pools cut each scan to `1/n`.
    """

    def __init__(self, transports: Sequence[httpx2.AsyncBaseTransport]) -> None:
        self.transports = list(transports)
        self._turn = itertools.cycle(self.transports)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await next(self._turn).handle_async_request(request)

    async def aclose(self) -> None:
        await asyncio.gather(*(transport.aclose() for transport in self.transports))


def engine_client(base_url: str, pools: int = 16) -> httpx2.AsyncClient:
    """An HTTP client for the inference engine at `base_url`.

    No timeout: a generation's duration is unbounded. No connection cap: a rollout's turns arrive together.
    """
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=None)
    transport = ShardedTransport([httpx2.AsyncHTTPTransport(limits=limits) for _ in range(pools)])
    return httpx2.AsyncClient(base_url=base_url, timeout=None, transport=transport)
