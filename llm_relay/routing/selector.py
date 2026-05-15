"""Model selection logic: filter, rank, pick."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config.loader import ConfigLoader
from ..config.types import ModelStatus, Privacy
from ..discovery.manager import DiscoveryManager


def _is_available(status: ModelStatus) -> bool:
    return status in (ModelStatus.available, ModelStatus.degraded)


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

    def select_best(self, ctx: RoutingContext) -> str | None:
        candidates, ordered = self._build_candidates(ctx)
        ctx.candidates = candidates
        filtered = self._apply_constraints(ctx, candidates)
        ctx.filtered = filtered
        if not filtered:
            return None
        if ordered:
            # Caller specified priority — honor it; only availability decides.
            ranked = list(filtered)
        else:
            ranked = self._rank(ctx, filtered)
        ctx.ranked = ranked
        # Walk in priority order and return the first that is currently up.
        for name in ranked:
            if _is_available(self.discovery.get_model_state(name)):
                return name
        return None

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
