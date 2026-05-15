"""CLI for llm-relay introspection."""
from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table

from .config.loader import ConfigLoader
from .config.types import Privacy
from .discovery.manager import DiscoveryManager
from .routing.selector import ModelSelector, RoutingContext


def _load_config() -> ConfigLoader:
    config_dir = os.environ.get("LLM_RELAY_CONFIG_DIR", "config")
    config = ConfigLoader(config_dir=config_dir)
    config.load()
    return config


def cmd_run(args: argparse.Namespace) -> int:
    import uvicorn

    port = args.port or int(os.environ.get("LLM_RELAY_PORT", 8090))
    host = args.host or os.environ.get("LLM_RELAY_HOST", "127.0.0.1")
    uvicorn.run(
        "llm_relay.api.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=args.reload,
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    config = _load_config()
    console = Console()
    table = Table(title="Configured Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider")
    table.add_column("Port")
    table.add_column("Class")
    table.add_column("Tags")
    for name, m in config.models.models.items():
        table.add_row(name, m.provider, str(m.port or "-"), m.class_name, ", ".join(m.tags))
    console.print(table)
    if config.models.aliases:
        atable = Table(title="Aliases")
        atable.add_column("Alias", style="magenta")
        atable.add_column("Candidates")
        for a, members in config.models.aliases.items():
            atable.add_row(a, ", ".join(members))
        console.print(atable)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    config = _load_config()
    console = Console()
    name = args.model
    if name in config.models.aliases:
        candidates = config.models.aliases[name]
        console.print(f"[yellow]Alias:[/yellow] {name}")
        console.print(f"[dim]Candidates:[/dim] {', '.join(candidates)}")
        return 0
    if name in config.models.models:
        m = config.models.models[name]
        console.print(f"[green]Model:[/green] {name}")
        console.print(f"[dim]Provider:[/dim] {m.provider}  [dim]Port:[/dim] {m.port or '-'}")
        return 0
    console.print(f"[red]Unknown:[/red] {name}")
    return 1


def cmd_health(args: argparse.Namespace) -> int:
    config = _load_config()
    console = Console()
    table = Table(title="Providers")
    table.add_column("Provider")
    table.add_column("Base URL")
    table.add_column("Status")
    for name, p in config.providers.items():
        table.add_row(name, p.base_url, "[green]enabled[/green]" if p.enabled else "[dim]disabled[/dim]")
    console.print(table)
    console.print("\n[dim]Live status: GET /health on the running service.[/dim]")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    config = _load_config()
    console = Console()
    discovery = DiscoveryManager()
    selector = ModelSelector(config, discovery)
    ctx = RoutingContext(
        requested_model=args.model,
        privacy=Privacy(args.privacy or "local_only"),
    )
    candidates, ordered = selector._build_candidates(ctx)
    filtered = selector._apply_constraints(ctx, candidates)
    ranked = list(filtered) if ordered else selector._rank(ctx, filtered)
    console.print(f"[bold]Routing simulation (no live availability):[/bold]")
    console.print(f"  requested: {args.model}")
    console.print(f"  candidates: {', '.join(candidates) or '(none)'}")
    console.print(f"  filtered:   {', '.join(filtered) or '(none)'}")
    console.print(f"  ranked:     {', '.join(ranked) or '(none)'}")
    return 0 if ranked else 1


def cmd_config(args: argparse.Namespace) -> int:
    config = _load_config()
    console = Console()
    console.print("[bold]Providers:[/bold]")
    for name, p in config.providers.items():
        console.print(f"  {name}: base_url={p.base_url} enabled={p.enabled}")
    console.print("\n[bold]Fallback graph:[/bold]")
    for key, chain in config.policy.fallback.graph.items():
        console.print(f"  {key}: {', '.join(chain)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="llm-relay", description="LLM relay routing control plane")
    subparsers = parser.add_subparsers(dest="command")

    p_run = subparsers.add_parser("run", help="Start the HTTP server")
    p_run.add_argument("--host")
    p_run.add_argument("--port", type=int)
    p_run.add_argument("--reload", action="store_true")

    p_models = subparsers.add_parser("models", help="Show configured models and aliases")
    p_models.add_argument("--available", action="store_true")

    p_resolve = subparsers.add_parser("resolve", help="Resolve a model name or alias")
    p_resolve.add_argument("model")

    subparsers.add_parser("health", help="Show provider config (live: GET /health)")

    p_route = subparsers.add_parser("route", help="Simulate a routing decision")
    p_route.add_argument("model")
    p_route.add_argument("--privacy", choices=["local_only", "cloud_ok"], default="local_only")

    subparsers.add_parser("config", help="Print loaded configuration")

    args = parser.parse_args()
    if args.command == "run":
        return cmd_run(args)
    if args.command == "models":
        return cmd_models(args)
    if args.command == "resolve":
        return cmd_resolve(args)
    if args.command == "health":
        return cmd_health(args)
    if args.command == "route":
        return cmd_route(args)
    if args.command == "config":
        return cmd_config(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
