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


def _insecure_bind_warning(host: str, auth_enabled: bool) -> str | None:
    """Warn when binding to a non-loopback interface with auth disabled (an open
    proxy to your models and backend topology). None when the bind is safe."""
    loopback = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}
    if host not in loopback and not auth_enabled:
        return (
            f"binding to {host} with auth DISABLED: anyone who can reach this port "
            "has an open proxy to your models and your backend topology. Set "
            "LLM_RELAY_AUTH=1 and mint keys with `llm-relay keys add` before exposing it."
        )
    return None


def cmd_run(args: argparse.Namespace) -> int:
    port = args.port or int(os.environ.get("LLM_RELAY_PORT", 8090))
    host = args.host or os.environ.get("LLM_RELAY_HOST", "127.0.0.1")
    try:
        auth_enabled = _load_config().auth.enabled
    except Exception:
        auth_enabled = False
    warning = _insecure_bind_warning(host, auth_enabled)
    if warning:
        Console(stderr=True).print(f"[bold red]WARNING:[/bold red] {warning}")
    if args.reload:
        # Dev mode: uvicorn's reloader needs an import string; single listener.
        import uvicorn

        uvicorn.run(
            "llm_relay.api.app:create_app",
            host=host,
            port=port,
            factory=True,
            reload=True,
        )
        return 0
    # Production path: one process, one or two listeners (LLM_RELAY_AUTH_PORT).
    import asyncio

    from .api.app import serve

    os.environ["LLM_RELAY_HOST"] = host
    os.environ["LLM_RELAY_PORT"] = str(port)
    asyncio.run(serve())
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
    ranked = list(filtered) if ordered else selector._rank(filtered)
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


def cmd_keys(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .audit import audit
    from .auth import add_key_record, load_key_records, revoke_hash, revoke_id

    config_dir = Path(os.environ.get("LLM_RELAY_CONFIG_DIR", "config"))
    path = config_dir / "api_keys.yaml"
    console = Console()
    action = getattr(args, "keys_action", None)
    if action == "add":
        config_dir.mkdir(parents=True, exist_ok=True)
        plaintext = add_key_record(
            path, args.id, priority_weight=args.priority, scopes=args.scopes,
            note=getattr(args, "note", None),
        )
        audit("key_minted", principal=args.id, scopes=args.scopes, by="cli")
        console.print(
            f"[green]Key for {args.id}[/green] (store securely, shown once): [bold]{plaintext}[/bold]"
        )
        return 0
    if action == "list":
        records = load_key_records(path)
        table = Table(title="API key principals")
        table.add_column("hash", style="dim")
        table.add_column("id", style="cyan")
        table.add_column("priority")
        table.add_column("scopes")
        table.add_column("enabled")
        table.add_column("created")
        table.add_column("note")
        for h, r in sorted(records.items()):
            table.add_row(
                h[:12], str(r.get("id", "?")), str(r.get("priority_weight", 1.0)),
                ", ".join(r.get("scopes") or []) or "-", str(r.get("enabled", True)),
                str(r.get("created", "")), str(r.get("note", "")),
            )
        console.print(table)
        return 0
    if action == "revoke":
        hash_prefix = getattr(args, "hash_prefix", None)
        if hash_prefix:
            n = revoke_hash(path, hash_prefix)
            if n == 1:
                audit("key_revoked", hash_prefix=hash_prefix, by="cli")
                console.print(f"[yellow]Revoked key {hash_prefix}...[/yellow]")
                return 0
            console.print(
                "[red]Prefix ambiguous; use a longer one[/red]" if n == -1
                else "[red]No key matches that prefix[/red]"
            )
            return 1
        if not args.id:
            console.print("[red]Give a principal id or --hash <prefix>[/red]")
            return 1
        removed = revoke_id(path, args.id)
        audit("key_revoked", principal=args.id, count=removed, by="cli")
        console.print(f"[yellow]Revoked {removed} key(s) for {args.id}[/yellow]")
        return 0
    console.print("[red]Usage: llm-relay keys {add|list|revoke}[/red]")
    return 1


def cmd_usage_backfill(args: argparse.Namespace) -> int:
    """Recover the token history Prometheus and the KPI store still hold.

    Safe to re-run: synthetic request ids are deterministic, so a second pass
    inserts nothing rather than doubling the totals, and ``INSERT OR IGNORE``
    means a backfill row can never overwrite a live per-request row.
    """
    from .usage_backfill import (
        apply_request_counts,
        backfill,
        day_range,
        prometheus_day_request_counts,
        prometheus_day_rows,
        rows_from_kpi_file,
    )
    from .usage_store import open_db

    console = Console()
    db_path = args.usage_db or os.environ.get("LLM_RELAY_USAGE_DB", "").strip()
    if not db_path and not args.dry_run:
        console.print(
            "[red]No usage database:[/red] pass --usage-db or set LLM_RELAY_USAGE_DB "
            "(or use --dry-run to only count)."
        )
        return 1
    if not args.prom_from and not args.kpi_file:
        console.print(
            "[red]Nothing to do:[/red] give --prom-from/--prom-to, --kpi-file, or both."
        )
        return 1
    if args.prom_from and not args.prom_to:
        console.print("[red]--prom-from requires --prom-to[/red]")
        return 1

    batches: list[tuple[str, list]] = []
    if args.prom_from:
        for day in day_range(args.prom_from, args.prom_to):
            try:
                rows = prometheus_day_rows(args.prom_url, day)
            except Exception as exc:
                # A skipped day is an invisible hole in the history, so stop
                # loudly instead of leaving one behind.
                console.print(f"[red]prometheus {day} failed:[/red] {exc}")
                return 1
            # Request counts are a second query and a softer failure: without
            # them each synthetic row stays at its floor of 1, which is wrong
            # but visibly wrong. Losing the tokens would be worse, so a count
            # gap is reported and the day is still written.
            models = {str(r.get("model") or "") for r in rows}
            try:
                counts = prometheus_day_request_counts(args.prom_url, day)
            except Exception as exc:
                console.print(
                    f"[yellow]{day}: per-model request counts unavailable "
                    f"({exc}) — request_count left at 1 for "
                    f"{len(models)} model(s)[/yellow]"
                )
            else:
                applied = apply_request_counts(rows, counts)
                missing = sorted(m for m in models - set(counts) if m)
                if missing:
                    console.print(
                        f"[yellow]{day}: no request count for "
                        f"{', '.join(missing)} — left at 1 "
                        f"({applied} model(s) counted)[/yellow]"
                    )
            batches.append((f"prom {day}", rows))
    if args.kpi_file:
        # Fleet-level KPI days must stop where the per-user Prometheus data
        # starts, or the same tokens get counted twice under two sources.
        cutover = args.kpi_before or args.prom_from
        if not cutover:
            console.print("[red]--kpi-file needs --kpi-before (or --prom-from)[/red]")
            return 1
        rows = rows_from_kpi_file(args.kpi_file, before_day=cutover)
        batches.append((f"kpi < {cutover}", rows))

    conn = None if args.dry_run else open_db(db_path)
    total_rows = 0
    total_inserted = 0
    try:
        for label, rows in batches:
            total_rows += len(rows)
            if conn is None:
                console.print(f"{label}: {len(rows)} row(s) [dim](dry run)[/dim]")
                continue
            inserted = backfill(conn, rows)
            total_inserted += inserted
            console.print(f"{label}: {len(rows)} row(s), {inserted} inserted")
    finally:
        if conn is not None:
            conn.close()
    if args.dry_run:
        console.print(f"[bold]Total:[/bold] {total_rows} row(s), nothing written")
    else:
        console.print(
            f"[bold]Total:[/bold] {total_rows} row(s), {total_inserted} inserted "
            f"({total_rows - total_inserted} already present)"
        )
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

    p_keys = subparsers.add_parser("keys", help="Manage per-user API keys")
    keys_sub = p_keys.add_subparsers(dest="keys_action")
    k_add = keys_sub.add_parser("add", help="Mint a new key for a user/agent")
    k_add.add_argument("id")
    k_add.add_argument("--priority", type=float, default=1.0)
    k_add.add_argument("--scope", action="append", default=[], dest="scopes")
    k_add.add_argument("--note", default=None)
    keys_sub.add_parser("list", help="List key principals (never prints keys)")
    k_rev = keys_sub.add_parser("revoke", help="Revoke keys by principal id or --hash prefix")
    k_rev.add_argument("id", nargs="?")
    k_rev.add_argument("--hash", dest="hash_prefix", default=None)

    p_backfill = subparsers.add_parser(
        "usage-backfill", help="Recover historical token usage into the usage store"
    )
    p_backfill.add_argument(
        "--usage-db", default=None,
        help="SQLite usage database (default: $LLM_RELAY_USAGE_DB)",
    )
    p_backfill.add_argument(
        "--prom-url", default="http://127.0.0.1:9090", help="Prometheus base URL"
    )
    p_backfill.add_argument("--prom-from", default=None, help="First day, YYYY-MM-DD")
    p_backfill.add_argument("--prom-to", default=None, help="Last day, YYYY-MM-DD")
    p_backfill.add_argument(
        "--kpi-file", default=None, help="Portal KPI JSONL with fleet daily totals"
    )
    p_backfill.add_argument(
        "--kpi-before", default=None,
        help="Only KPI days strictly before this YYYY-MM-DD (the Prometheus cutover)",
    )
    p_backfill.add_argument(
        "--dry-run", action="store_true", help="Print counts without writing"
    )

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
    if args.command == "keys":
        return cmd_keys(args)
    if args.command == "usage-backfill":
        return cmd_usage_backfill(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
