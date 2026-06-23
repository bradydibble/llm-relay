"""YAML configuration loader for llm-relay."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml

from .types import (
    CategoryConfig,
    CircuitBreaker,
    ExplicitBehavior,
    FallbackGraph,
    ModelConfig,
    ModeConfig,
    ModeHint,
    PolicyConfig,
    PrivacyConstraints,
    Privacy,
    ProviderConfig,
    ProviderType,
)
from ..auth import AuthConfig, load_keys

logger = logging.getLogger(__name__)


def _parse_duration(s) -> int:
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if s.endswith("s"):
        return int(s[:-1])
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    return int(s)


class ModelRegistry:
    """Concrete models + alias map. Accessed as config.models.models / config.models.aliases."""

    def __init__(self):
        self.models: dict[str, ModelConfig] = {}
        self.aliases: dict[str, list[str]] = {}
        self.categories: dict[str, CategoryConfig] = {}
        # Variant grouping (plan 2): logical_models[L] = [variant config names],
        # derived from each model's optional `logical` field at load.
        self.logical_models: dict[str, list[str]] = {}
        # Per-host exclusivity (plan 2): groups of model names that cannot be hot
        # at once because they share a served (provider, port). Derived at load.
        self.exclusivity_groups: list[list[str]] = []

    def variants_of(self, logical: str) -> list[str]:
        """Variant config names grouped under a logical model (empty if none)."""
        return list(self.logical_models.get(logical, []))

    def logical_of(self, name: str) -> str | None:
        """The logical model a config entry is a variant of, or None."""
        m = self.models.get(name)
        return m.logical if m else None

    def exclusive_with(self, name: str) -> list[str]:
        """Model names mutually exclusive with ``name`` (they share its served
        provider+port), excluding itself. Empty when ``name`` shares no port."""
        for group in self.exclusivity_groups:
            if name in group:
                return [n for n in group if n != name]
        return []


class ConfigLoader:
    def __init__(self, config_dir: Path | str | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._providers: dict[str, ProviderConfig] = {}
        self._models = ModelRegistry()
        self._modes: dict[str, ModeConfig] = {}
        self._policy: PolicyConfig | None = None
        # Always present so middleware reading config.auth never hits an
        # AttributeError before load() runs; load() overwrites it.
        self.auth: AuthConfig = AuthConfig()

    def load(self) -> None:
        self._load_providers()
        self._load_models()
        self._load_modes()
        self._load_policy()
        self._load_auth()

    def _load_auth(self) -> None:
        """Load the per-user API-key store into ``self.auth``. Enabled by the
        ``LLM_RELAY_AUTH=1`` env flag or an ``auth.enabled: true`` block in an
        optional ``auth.yaml``; principals come from ``api_keys.yaml`` (kept
        off-repo in the config dir, never committed)."""
        auth_file = self.config_dir / "api_keys.yaml"
        auth_block: dict = {}
        auth_yaml = self.config_dir / "auth.yaml"
        if auth_yaml.exists():
            auth_block = (yaml.safe_load(auth_yaml.read_text()) or {}).get("auth", {}) or {}
        enabled = os.environ.get("LLM_RELAY_AUTH") == "1" or bool(auth_block.get("enabled", False))
        self.auth = AuthConfig(
            enabled=enabled,
            exempt_paths=list(auth_block.get("exempt_paths", ["/health", "/metrics"])),
            principals_by_hash=load_keys(auth_file),
        )

    # --- Maintenance-pause persistence (so a paused provider survives a relay
    # restart). {provider: {"until": str|null, "reason": str|null}} in
    # paused-providers.json under the config dir. ---
    def load_paused_providers(self) -> dict:
        path = self.config_dir / "paused-providers.json"
        try:
            with open(path) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_paused_providers(self, paused: dict) -> None:
        path = self.config_dir / "paused-providers.json"
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(paused, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _load_providers(self) -> None:
        path = self.config_dir / "providers.yaml"
        if not path.exists():
            return
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for name, cfg in (data.get("providers") or {}).items():
            cb = cfg.get("circuit_breaker") or {}
            self._providers[name] = ProviderConfig(
                type=ProviderType(cfg.get("type", "openai")),
                base_url=cfg["base_url"],
                enabled=cfg.get("enabled", True),
                auth_source=cfg.get("auth_source"),
                health_endpoint=cfg.get("health_endpoint", "/v1/models"),
                poll_interval=_parse_duration(cfg.get("poll_interval", "15s")),
                health_check_timeout=_parse_duration(cfg.get("health_check_timeout", "5s")),
                circuit_breaker=CircuitBreaker(
                    failure_threshold=cb.get("failure_threshold", 3),
                    recovery_timeout=_parse_duration(cb.get("recovery_timeout", 30)),
                ),
                model_overrides=cfg.get("model_overrides", []) or [],
                max_concurrent=cfg.get("max_concurrent"),
                slot_wait_timeout=float(cfg.get("slot_wait_timeout", 30.0)),
                discover_ports=cfg.get("discover_ports", []) or [],
            )

    def _load_models(self) -> None:
        path = self.config_dir / "models.yaml"
        if not path.exists():
            return
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        models_data = data.get("models") or {}
        for name, cfg in models_data.items():
            if name == "aliases":
                logger.warning(
                    "an explicit `aliases:` block in models.yaml is no longer supported "
                    "and was ignored; categories are derived from per-model `use_cases` "
                    "tags. Remove the block and tag the models instead."
                )
                continue
            if name == "categories":
                for cat_name, meta in (cfg or {}).items():
                    self._models.categories[cat_name] = CategoryConfig(
                        reasoning_floor=(meta or {}).get("reasoning_floor"),
                    )
                continue
            self._models.models[name] = ModelConfig(
                provider=cfg["provider"],
                class_name=cfg.get("class", "unknown"),
                port=cfg.get("port"),
                path=cfg.get("path", "") or "",
                service=cfg.get("service"),
                served_model_name=cfg.get("served_model_name"),
                context_window=cfg.get("context_window"),
                capabilities=cfg.get("capabilities", []) or [],
                tags=cfg.get("tags", []) or [],
                preference=cfg.get("preference", 0.5),
                privacy=Privacy(cfg.get("privacy", "local_only")),
                use_cases={k: float(v) for k, v in (cfg.get("use_cases") or {}).items()},
                manual_only=bool(cfg.get("manual_only", False)),
                logical=cfg.get("logical"),
                quant=cfg.get("quant"),
            )
        self._derive_aliases_from_use_cases()
        self._derive_logical_models()
        self._derive_exclusivity()

    def _derive_aliases_from_use_cases(self) -> None:
        """Transpose per-model ``use_cases`` tags into the alias map: for each
        use-case, ``aliases[uc] = models tagged uc, sorted by (uc-priority desc,
        preference desc, name asc)`` — the same ordering the selector and MCP
        ``select_for_capability`` use, so the surfaces never disagree.

        Categories are derived purely from tags — there is no static ``aliases:``
        block any more (an explicit block is ignored at load; see ``_load_models``).
        """
        derived: dict[str, list[str]] = {}
        for mname, mcfg in self._models.models.items():
            for uc in mcfg.use_cases:
                derived.setdefault(uc, []).append(mname)
        for uc, names in derived.items():
            names.sort(key=lambda n: (
                -(self._models.models[n].use_cases.get(uc) or 0.0),
                -(self._models.models[n].preference or 0.0),
                n,
            ))
            self._models.aliases[uc] = names

    def _derive_logical_models(self) -> None:
        """Group variants by their ``logical`` field:
        ``logical_models[L] = sorted [variant config names]``. Entries with no
        ``logical`` are standalone models and are not grouped. Additive: the flat
        ``models`` dict is unchanged, so existing routing is unaffected."""
        derived: dict[str, list[str]] = {}
        for name, m in self._models.models.items():
            if m.logical:
                derived.setdefault(m.logical, []).append(name)
        for names in derived.values():
            names.sort()
        self._models.logical_models = derived

    def _derive_exclusivity(self) -> None:
        """Models sharing the same served ``(provider, port)`` are mutually
        exclusive: one served instance per port. Derive groups of size > 1.
        ``port=None`` entries (e.g. cloud) are never grouped, since they serve
        concurrently rather than swapping on a physical port."""
        by_port: dict[tuple[str, int], list[str]] = {}
        for name, m in self._models.models.items():
            if m.port is None:
                continue
            by_port.setdefault((m.provider, m.port), []).append(name)
        self._models.exclusivity_groups = [
            sorted(names) for names in by_port.values() if len(names) > 1
        ]

    def _load_modes(self) -> None:
        path = self.config_dir / "modes.yaml"
        if not path.exists():
            return
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for name, cfg in (data.get("modes") or {}).items():
            self._modes[name] = ModeConfig(
                description=cfg.get("description", ""),
                ports=cfg.get("ports", []) or [],
                models=cfg.get("models", []) or [],
                default=cfg.get("default", ""),
            )

    def _load_policy(self) -> None:
        path = self.config_dir / "policy.yaml"
        if not path.exists():
            self._policy = PolicyConfig()
            return
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        policy_data = data.get("policy") or {}
        constraints = policy_data.get("constraints") or {}
        priv = constraints.get("privacy") or {}
        fallback = policy_data.get("fallback") or {}
        self._policy = PolicyConfig(
            constraints=PrivacyConstraints(
                default=Privacy(priv.get("default", "local_only")),
                cloud_allowed_tags=priv.get("cloud_allowed_tags", []) or [],
            ),
            fallback=FallbackGraph(
                graph=fallback.get("graph", {}) or {},
                retry_on=fallback.get("retry_on", ["502", "503", "504", "connection_error"]) or [],
            ),
            explicit=ExplicitBehavior(
                strict=(policy_data.get("explicit") or {}).get("strict", False),
            ),
            mode_hints=[
                ModeHint(
                    when_requesting=h["when_requesting"],
                    unavailable_action=h["unavailable_action"],
                    recommended_mode=h.get("recommended_mode"),
                    alternative=h.get("alternative"),
                    message=h.get("message", ""),
                )
                for h in (policy_data.get("mode_hints") or [])
            ],
        )

    @property
    def providers(self) -> dict[str, ProviderConfig]:
        return self._providers

    @property
    def models(self) -> ModelRegistry:
        return self._models

    @property
    def modes(self) -> dict[str, ModeConfig]:
        return self._modes

    @property
    def policy(self) -> PolicyConfig:
        if self._policy is None:
            self._policy = PolicyConfig()
        return self._policy

    def get_provider(self, name: str) -> ProviderConfig | None:
        return self._providers.get(name)

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.models.get(name)
