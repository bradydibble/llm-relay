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
            # Open-by-default: an alias is a PRIORITY ORDER over the whole live
            # fleet, not a whitelist. Named members are the priority prefix; every
            # other live model follows as a preference-ranked tail, so the alias
            # degrades to anything live rather than dead-ending when its members
            # are down. Context / privacy / reasoning-floor filters run downstream
            # in _apply_constraints, so the tail only yields a model that can serve.
            # Fallthrough is for aliases only — explicit / host-pinned requests
            # (below) stay deliberately specific.
            members = self.resolve_alias(req)
            return members + self._fleet_tail(members), True
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

    def _apply_constraints(
        self, ctx: RoutingContext, candidates: list[str], *, ignore_context: bool = False
    ) -> list[str]:
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
            # `ignore_context` skips this gate so callers can ask "what passes the
            # NON-context constraints?" (used to diagnose a context shortfall).
            ctx_window = self.discovery.get_live_context_window(name) or cfg.context_window
            if not ignore_context and ctx.min_context and ctx_window and ctx_window < ctx.min_context:
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

    def _fleet_tail(self, exclude: list[str]) -> list[str]:
        """The open-by-default fallthrough tail: every live model NOT already named
        in ``exclude``, preference-ranked.

        Sourced from discovery (the live fleet) — the same surface the unknown-model
        branch uses — so an alias degrades to whatever is currently up. Availability,
        context-fit, privacy and the reasoning floor are all enforced later in
        ``_apply_constraints`` / selection, so this only widens the candidate set; it
        never admits a model that cannot actually serve the request.
        """
        excluded = set(exclude)
        return self._rank([m for m in self.discovery.get_available_models() if m not in excluded])

    def _window_of(self, name: str) -> int:
        """Effective context window for `name`: live max_model_len when a backend
        reports one, else the static config window (0 if unknown)."""
        cfg = self.config.models.models.get(name)
        if cfg is None:
            return 0
        return self.discovery.get_live_context_window(name) or (cfg.context_window or 0)

    def diagnose_context_shortfall(self, ctx: RoutingContext) -> dict | None:
        """Explain a no-candidate outcome when the binding constraint is context.

        Returns a structured, actionable diagnosis when the request's estimated
        context exceeds every admissible LIVE model's window, distinguishing:

          * ``oversize_for_now`` — a big-enough model exists in the catalog but is
            currently down, so waiting (back off + retry) can succeed; vs.
          * ``oversize_period`` — nothing in the whole admissible catalog is big
            enough, so only resizing or deferring the request can.

        Returns ``None`` when context is NOT the binding constraint: the request
        fits some live model, or nothing is live at all (a plain availability
        outage the generic 503 already covers). The relay routes, it does not
        chunk — this lets a deterministic client size down or wait without any
        model-side (LLM) decision.
        """
        if not ctx.min_context:
            return None
        admissible = self._apply_constraints(ctx, list(self.config.models.models), ignore_context=True)
        max_live = max(
            (self._window_of(m) for m in admissible if _is_available(self.discovery.get_model_state(m))),
            default=0,
        )
        if max_live == 0 or ctx.min_context <= max_live:
            return None
        max_catalog = max((self.config.models.models[m].context_window or 0 for m in admissible), default=0)
        return {
            "reason": "request_exceeds_live_context",
            "estimated_tokens": ctx.min_context,
            "max_available_now": max_live,
            "max_in_catalog": max_catalog,
            "classification": "oversize_for_now" if ctx.min_context <= max_catalog else "oversize_period",
        }

    def get_fallback_chain(self, model_name: str) -> list[str]:
        for chain in self.config.policy.fallback.graph.values():
            if model_name in chain:
                return [m for m in chain if self.discovery.get_model_state(m) != ModelStatus.unavailable]
        if self.discovery.get_model_state(model_name) == ModelStatus.available:
            return [model_name]
        return []
