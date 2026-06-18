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


class ConfigLoader:
    def __init__(self, config_dir: Path | str | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._providers: dict[str, ProviderConfig] = {}
        self._models = ModelRegistry()
        self._modes: dict[str, ModeConfig] = {}
        self._policy: PolicyConfig | None = None

    def load(self) -> None:
        self._load_providers()
        self._load_models()
        self._load_modes()
        self._load_policy()

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
            )
        self._derive_aliases_from_use_cases()

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
