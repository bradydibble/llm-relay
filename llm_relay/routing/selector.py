"""Model selection logic: filter, rank, pick."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config.loader import ConfigLoader
from ..config.types import ModelStatus, Privacy
from ..discovery.manager import DiscoveryManager


def _is_available(status: ModelStatus) -> bool:
    return status in (ModelStatus.available, ModelStatus.degraded)


@dataclass
class ChainCandidate:
    """A fully-resolved candidate ready for forwarding.

    Carries all metadata that ``forward_request`` / ``stream_request`` need so
    the retry loop in ``route_and_forward`` can build a ``RouteResult`` for each
    hop without touching the config again.
    """

    model: str
    backend_url: str
    backend_key: str
    slot_wait_timeout: float
    provider_name: str


@dataclass
class RoutingContext:
    requested_model: str
    privacy: Privacy = Privacy.local_only
    weights: dict[str, float] = field(default_factory=dict)
    require_tools: bool = False
    min_context: int | None = None
    resolved_model: str | None = None
    candidates: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)
    ranked: list[str] = field(default_factory=list)


class ModelSelector:
    def __init__(self, config: ConfigLoader, discovery: DiscoveryManager):
        self.config = config
        self.discovery = discovery

    def is_alias(self, name: str) -> bool:
        return name in self.config.models.aliases

    def resolve_alias(self, name: str) -> list[str]:
        if self.is_alias(name):
            return list(self.config.models.aliases[name])
        return [name]

    def _prepare_ranked(self, ctx: RoutingContext) -> list[str]:
        """Build and store the full ranked candidate list on *ctx*.

        Populates ``ctx.candidates``, ``ctx.filtered``, and ``ctx.ranked``.
        Returns ``ctx.ranked``.  Called by both ``select_best`` and
        ``select_chain`` so they share exactly the same filter/rank logic.
        """
        candidates, ordered = self._build_candidates(ctx)
        ctx.candidates = candidates
        filtered = self._apply_constraints(ctx, candidates)
        ctx.filtered = filtered
        if not filtered:
            ctx.ranked = []
            return []
        if ordered:
            ranked = list(filtered)
        else:
            ranked = self._rank(ctx, filtered)
        ctx.ranked = ranked
        return ranked

    def select_best(self, ctx: RoutingContext) -> str | None:
        ranked = self._prepare_ranked(ctx)
        # Walk in priority order and return the first that is currently up.
        for name in ranked:
            if _is_available(self.discovery.get_model_state(name)):
                return name
        return None

    def select_chain(self, ctx: RoutingContext) -> list[ChainCandidate]:
        """Return every viable candidate in priority order as ``ChainCandidate`` objects.

        Unlike ``select_best`` (which returns only the first available), this
        returns all candidates that have a resolvable backend_url and a config
        entry — the retry loop in ``route_and_forward`` then decides which ones
        to try based on upstream responses.

        Only candidates that are currently available (or degraded) AND whose
        backend URL can be resolved are included; broken/unconfigured entries
        are silently skipped to keep the chain clean.
        """
        ranked = self._prepare_ranked(ctx)
        out: list[ChainCandidate] = []
        for name in ranked:
            if not _is_available(self.discovery.get_model_state(name)):
                continue
            cfg = self.config.models.models.get(name)
            if not cfg:
                continue
            provider = self.config.providers.get(cfg.provider)
            if not provider:
                continue
            # Build backend URL (mirrors RequestRouter._backend_url logic).
            url = provider.base_url.rstrip("/")
            if cfg.port:
                url = f"{url}:{cfg.port}"
            if cfg.path:
                url = f"{url}/{cfg.path.lstrip('/')}"
            backend_url = f"{url}/v1"
            # Build backend key (mirrors _compose_backend_key in router.py).
            key_parts = [cfg.provider]
            if cfg.port:
                key_parts.append(str(cfg.port))
            if cfg.path:
                key_parts.append(cfg.path.strip("/"))
            backend_key = ":".join(key_parts)
            slot_wait_timeout = provider.slot_wait_timeout if hasattr(provider, "slot_wait_timeout") else 30.0
            out.append(ChainCandidate(
                model=name,
                backend_url=backend_url,
                backend_key=backend_key,
                slot_wait_timeout=slot_wait_timeout,
                provider_name=cfg.provider,
            ))
        return out

    def _build_candidates(self, ctx: RoutingContext) -> tuple[list[str], bool]:
        """Return (candidates, ordered). When ordered=True, list order is the priority."""
        req = ctx.requested_model
        if self.is_alias(req):
            return self.resolve_alias(req), True
        if req in self.config.models.models:
            if self.config.policy.explicit.strict:
                return [req], True
            chain = self.config.policy.fallback.graph.get(req, [])
            seen: set[str] = set()
            out: list[str] = []
            for m in [req, *chain]:
                if m not in seen:
                    out.append(m)
                    seen.add(m)
            return out, True
        if req in self.config.policy.fallback.graph:
            return list(self.config.policy.fallback.graph[req]), True
        return list(self.discovery.get_available_models().keys()), False

    def _apply_constraints(self, ctx: RoutingContext, candidates: list[str]) -> list[str]:
        out: list[str] = []
        for name in candidates:
            cfg = self.config.models.models.get(name)
            if not cfg:
                continue
            if ctx.privacy == Privacy.local_only and cfg.privacy == Privacy.cloud_ok:
                continue
            if ctx.require_tools and "tool_use" not in cfg.capabilities:
                continue
            if ctx.min_context and cfg.context_window and cfg.context_window < ctx.min_context:
                continue
            out.append(name)
        return out

    def _rank(self, ctx: RoutingContext, candidates: list[str]) -> list[str]:
        w = ctx.weights or {
            "quality": self.config.policy.ranking.quality,
            "latency": self.config.policy.ranking.latency,
            "cost": self.config.policy.ranking.cost,
            "availability": self.config.policy.ranking.availability,
        }
        scored: list[tuple[str, float]] = []
        for name in candidates:
            cfg = self.config.models.models.get(name)
            if not cfg:
                continue
            q = cfg.preference * w.get("quality", 0.4)
            latency = (0.9 if "local" in cfg.tags else 0.5) * w.get("latency", 0.3)
            cost = (1.0 if "local" in cfg.tags else 0.3) * w.get("cost", 0.1)
            status = self.discovery.get_model_state(name)
            av = {
                ModelStatus.available: 1.0,
                ModelStatus.degraded: 0.5,
                ModelStatus.unavailable: 0.0,
            }.get(status, 0.0) * w.get("availability", 0.2)
            scored.append((name, q + latency + cost + av))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def get_fallback_chain(self, model_name: str) -> list[str]:
        for chain in self.config.policy.fallback.graph.values():
            if model_name in chain:
                return [m for m in chain if self.discovery.get_model_state(m) != ModelStatus.unavailable]
        if self.discovery.get_model_state(model_name) == ModelStatus.available:
            return [model_name]
        return []
