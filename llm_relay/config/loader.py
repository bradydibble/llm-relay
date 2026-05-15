"""YAML configuration loader for llm-relay."""
from __future__ import annotations

from pathlib import Path

import yaml

from .types import (
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
    RankingWeights,
)


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
                for alias_name, members in (cfg or {}).items():
                    self._models.aliases[alias_name] = list(members)
                continue
            self._models.models[name] = ModelConfig(
                provider=cfg["provider"],
                class_name=cfg.get("class", "unknown"),
                port=cfg.get("port"),
                path=cfg.get("path", "") or "",
                service=cfg.get("service"),
                context_window=cfg.get("context_window"),
                capabilities=cfg.get("capabilities", []) or [],
                tags=cfg.get("tags", []) or [],
                preference=cfg.get("preference", 0.5),
                privacy=Privacy(cfg.get("privacy", "local_only")),
            )

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
        ranking = policy_data.get("ranking") or {}
        constraints = policy_data.get("constraints") or {}
        priv = constraints.get("privacy") or {}
        fallback = policy_data.get("fallback") or {}
        self._policy = PolicyConfig(
            ranking=RankingWeights(
                quality=ranking.get("quality", 0.4),
                latency=ranking.get("latency", 0.3),
                cost=ranking.get("cost", 0.1),
                availability=ranking.get("availability", 0.2),
            ),
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
