"""Model selection logic: filter, rank, pick."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config.loader import ConfigLoader
from ..config.types import ModelStatus, Privacy
from ..discovery.manager import DiscoveryManager
from .keys import compose_backend_key, compose_backend_url, resolve_model_id


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
            ranked = self._rank(filtered)
        # Load-aware re-sort: prefer least-loaded backend, break ties on the
        # original priority order. Targets TTFT/TPS — even one in-flight slot
        # on the priority backend can add multi-second slot-wait latency, so
        # we aggressively spill to an idle alternate when one exists.
        #
        # Skip entirely when there's only one candidate (e.g. strict mode with
        # the requested model present): there's no load decision to make, and
        # this keeps a corrupt inflight counter on the lone backend from ever
        # perturbing the routing decision.
        if len(ranked) > 1:
            ranked = self._sort_by_load(ranked)
        ctx.ranked = ranked
        return ranked

    def _sort_by_load(self, ranked: list[str]) -> list[str]:
        """Re-order *ranked* so least-loaded candidates win; preserve original
        priority on ties.

        Sort key per candidate: ``(load_ratio, original_index)``. A candidate
        with no backend client or no semaphore (unbounded) scores ``load_ratio
        = 0.0`` — treated as fully idle.
        """
        scored: list[tuple[float, int, str]] = [
            (self._load_ratio(name), idx, name) for idx, name in enumerate(ranked)
        ]
        scored.sort()
        return [name for _, _, name in scored]

    def _load_ratio(self, model_name: str) -> float:
        """Current in-flight ratio for the backend serving *model_name*.

        Returns 0.0 for backends with no semaphore (max_concurrent unset) or
        when the model can't be resolved to a registered backend — those are
        treated as "no contention" so they don't get penalized in routing.
        """
        cfg = self.config.models.models.get(model_name)
        if not cfg:
            return 0.0
        key = compose_backend_key(cfg.provider, cfg.port, cfg.path or "")
        client = self.discovery.clients.get(key)
        if client is None or client.max_concurrent is None or client.max_concurrent <= 0:
            return 0.0
        return client.inflight_used / client.max_concurrent

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
            backend_url = compose_backend_url(provider.base_url, cfg.port, cfg.path)
            backend_key = compose_backend_key(cfg.provider, cfg.port, cfg.path or "")
            slot_wait_timeout = provider.slot_wait_timeout
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
        # Normalize a host-qualified 'provider:model' id to its bare config name
        # (the provider is validated against the model's configured provider).
        # A bare name or unknown id passes through unchanged.
        resolved = resolve_model_id(self.config.models.models, req)
        if resolved is not None:
            req = resolved
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
            # Filter on the LIVE context window when a backend reports one, so
            # routing never admits a request larger than what /v1/models
            # advertised (which is also live-preferred). Fall back to the static
            # config value only when no backend currently reports max_model_len.
            ctx_window = self.discovery.get_live_context_window(name) or cfg.context_window
            if ctx.min_context and ctx_window and ctx_window < ctx.min_context:
                continue
            out.append(name)
        return out

    def _rank(self, candidates: list[str]) -> list[str]:
        """Deterministic preference sort for the unknown-model branch.

        Orders by configured ``preference`` (descending), breaking ties on name
        (ascending) -- the same ordering MCP ``select_for_capability`` returns,
        so the two discovery surfaces never disagree. Candidates without a
        config entry are skipped (their preference is unknown), preserving the
        prior behaviour.
        """
        models = self.config.models.models
        configured = [name for name in candidates if name in models]
        configured.sort(key=lambda name: (-(models[name].preference or 0.0), name))
        return configured

    def get_fallback_chain(self, model_name: str) -> list[str]:
        for chain in self.config.policy.fallback.graph.values():
            if model_name in chain:
                return [m for m in chain if self.discovery.get_model_state(m) != ModelStatus.unavailable]
        if self.discovery.get_model_state(model_name) == ModelStatus.available:
            return [model_name]
        return []
