"""CLI for llm-relay introspection."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

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


def cmd_cache_sample(args: argparse.Namespace) -> int:
    """Sample each backend's prefix-cache counters into the usage database.

    Meant to run periodically (a timer, every few minutes). The counters are
    cumulative from backend process start, so a per-day number only exists as a
    delta against the persisted cursor: run #1 seeds baselines and attributes
    nothing, and a re-run of an unchanged counter adds nothing.

    Fleet-level per (day, model) only. Per-request cache attribution comes from
    ``prompt_tokens_details.cached_tokens`` on the request path instead, and the
    two are not expected to agree -- see ``cache_sampler`` for why.
    """
    from .cache_sampler import (
        Backend,
        backends_from_config,
        cache_rollup,
        fetch_metrics,
        open_cache_db,
        parse_prefix_cache_metrics,
        sample_backends,
        utc_day,
    )

    console = Console()
    if args.backend:
        # Explicit targets, for a backend the relay does not (yet) route to.
        backends = [
            Backend(key=url, provider="cli", base_url=url.rstrip("/"))
            for url in args.backend
        ]
    else:
        try:
            backends = backends_from_config(_load_config())
        except Exception as exc:
            console.print(f"[red]Cannot load config:[/red] {exc}")
            return 1
    if not backends:
        console.print("[red]No backends:[/red] pass --backend URL or configure a provider")
        return 1

    day = args.day or utc_day()
    if args.dry_run:
        # Show what each backend reports without touching the database, so the
        # scrape can be verified before it starts moving cursors.
        table = Table(title=f"Prefix cache (dry run, {day})")
        for column in ("Backend", "Model", "Queried", "Hits", "Rate"):
            table.add_column(column)
        for backend in backends:
            try:
                reading = parse_prefix_cache_metrics(
                    fetch_metrics(backend.base_url, timeout=args.timeout)
                )
            except Exception as exc:
                table.add_row(backend.key, "[red]unreachable[/red]", "-", "-", str(exc))
                continue
            if not reading.reported:
                table.add_row(backend.key, "[dim]not reported[/dim]", "-", "-", "-")
                continue
            for model, counters in reading.by_model.items():
                rate = (f"{counters.hits / counters.queried:.1%}"
                        if counters.queried else "-")
                table.add_row(backend.key, model or "(unlabelled)",
                              str(counters.queried), str(counters.hits), rate)
        console.print(table)
        return 0

    db_path = args.usage_db or os.environ.get("LLM_RELAY_USAGE_DB", "").strip()
    if not db_path:
        console.print(
            "[red]No usage database:[/red] pass --usage-db or set LLM_RELAY_USAGE_DB "
            "(or use --dry-run to only read)."
        )
        return 1

    conn = open_cache_db(db_path)
    try:
        result = sample_backends(conn, backends, day=day, timeout=args.timeout)
        rows = cache_rollup(conn, day, day)
    finally:
        conn.close()

    console.print(
        f"[bold]{day}:[/bold] {result.counted} counted, {result.baselined} baselined, "
        f"{result.unchanged} unchanged, {result.resets} reset(s), "
        f"{result.rejected} rejected"
    )
    if result.unreachable:
        console.print(f"[yellow]unreachable:[/yellow] {', '.join(result.unreachable)}")
    if result.not_reported:
        # Not the same as zero reuse: these backends expose no prefix-cache
        # series (not vLLM, or caching off, or the metric renamed).
        console.print(f"[dim]no prefix-cache metrics:[/dim] {', '.join(result.not_reported)}")
    if result.unattributed:
        console.print(
            f"[yellow]unlabelled metrics on a multi-model backend:[/yellow] "
            f"{', '.join(result.unattributed)}"
        )
    table = Table(title=f"Prefix cache reuse ({day})")
    table.add_column("Model", style="cyan")
    table.add_column("Queried", justify="right")
    table.add_column("Cache read", justify="right")
    table.add_column("Hit rate", justify="right")
    for row in rows:
        table.add_row(
            row["model"], f"{row['queried_tokens']:,}",
            f"{row['cache_read_tokens']:,}",
            f"{row['hit_rate']:.1%}" if row["hit_rate"] is not None
            else ("[dim]not reported[/dim]" if not row["reported"] else "-"),
        )
    console.print(table)
    return 0


def _prompt_db_path(args: argparse.Namespace) -> str:
    return (getattr(args, "db", None) or os.environ.get("LLM_RELAY_PROMPT_DB", "")).strip()


def _parse_day(raw: str):
    """A YYYY-MM-DD day, or None if it is not one."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _prune_preview(conn, cutoff: str) -> dict:
    """What a prune at ``cutoff`` would delete, without writing anything.

    Mirrors ``prompt_store.prune``'s rule exactly: a request goes when its day
    is before the cutoff, and a stored message goes only when NO surviving
    request still references it -- content addressing means one message can be
    shared by thousands of requests, and a shared system prompt must not vanish
    because the oldest conversation using it aged out. Already-orphaned rows are
    counted here because prune deletes those too.

    Lives in the CLI rather than the store so the preview path is read-only by
    construction; a test pins the two to the same counts on one fixture.
    """
    requests = conn.execute(
        "SELECT COUNT(*) FROM prompt_requests WHERE day < ?", (cutoff,)
    ).fetchone()[0]
    messages = conn.execute(
        "SELECT COUNT(*) FROM messages m WHERE NOT EXISTS ("
        " SELECT 1 FROM request_messages rm"
        " JOIN prompt_requests pr ON pr.request_id = rm.request_id"
        " WHERE rm.message_id = m.id AND pr.day >= ?)",
        (cutoff,),
    ).fetchone()[0]
    return {"requests": int(requests), "messages": int(messages)}


def cmd_prompts_prune(args: argparse.Namespace) -> int:
    """Delete captured prompt content older than a cutoff day.

    Retention is indefinite by decision, so this does NOTHING unless a cutoff
    is named: either ``--older-than YYYY-MM-DD``, or
    ``LLM_RELAY_PROMPT_RETENTION_DAYS`` set to a positive number of days. Unset
    -- or 0 -- means keep forever, and that is the behavior with no
    configuration at all. The prune path ships so the retention choice stays
    revisitable on evidence, not so it quietly starts deleting.
    """
    from .audit import audit
    from .prompt_store import open_db, prune

    console = Console()
    cutoff = (getattr(args, "older_than", None) or "").strip()
    if cutoff:
        if _parse_day(cutoff) is None:
            console.print(
                f"[red]Invalid --older-than:[/red] {cutoff!r} is not a YYYY-MM-DD date"
            )
            return 1
    else:
        raw = os.environ.get("LLM_RELAY_PROMPT_RETENTION_DAYS", "").strip()
        if not raw:
            console.print(
                "Retention is indefinite (LLM_RELAY_PROMPT_RETENTION_DAYS unset): "
                "nothing pruned. Pass --older-than YYYY-MM-DD to prune anyway."
            )
            return 0
        try:
            days = int(raw)
        except ValueError:
            console.print(
                f"[red]Invalid LLM_RELAY_PROMPT_RETENTION_DAYS:[/red] {raw!r} is not "
                "a whole number of days"
            )
            return 1
        if days < 0:
            # Not silently treated as "keep forever": a negative window is a
            # typo, and a typo that deletes nothing while looking configured is
            # how a retention policy quietly stops existing.
            console.print(
                f"[red]Invalid LLM_RELAY_PROMPT_RETENTION_DAYS:[/red] {days} days is "
                "negative; use 0 (or unset) to keep forever"
            )
            return 1
        if days == 0:
            console.print(
                "Retention is indefinite (LLM_RELAY_PROMPT_RETENTION_DAYS=0): "
                "nothing pruned. Pass --older-than YYYY-MM-DD to prune anyway."
            )
            return 0
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        console.print(f"[dim]retention {days} day(s) -> cutoff {cutoff}[/dim]")

    path = _prompt_db_path(args)
    if not path:
        console.print(
            "[red]No prompt database:[/red] pass --db PATH or set LLM_RELAY_PROMPT_DB"
        )
        return 1
    if not os.path.exists(path):
        # Never create a store as a side effect of pruning one.
        console.print(f"No prompt store at {path}: nothing to prune.")
        return 0

    conn = open_db(path)
    try:
        if args.dry_run:
            counts = _prune_preview(conn, cutoff)
            console.print(
                f"[bold]dry run, cutoff {cutoff}:[/bold] would delete "
                f"{counts['requests']} request(s) and {counts['messages']} stored "
                "message(s); nothing written"
            )
            return 0
        result = prune(conn, cutoff)
    finally:
        conn.close()

    audit(
        "prompt_prune", cutoff=cutoff, by="cli",
        requests_deleted=result["requests_deleted"],
        messages_deleted=result["messages_deleted"],
        index_stale=result["index_stale"],
    )
    console.print(
        f"[bold]cutoff {cutoff}:[/bold] deleted {result['requests_deleted']} "
        f"request(s) and {result['messages_deleted']} stored message(s)"
    )
    if result["index_stale"]:
        console.print(
            f"[yellow]{result['index_stale']} search-index row(s) could not be "
            "removed[/yellow] (unreadable blob); they resolve to no content."
        )
    return 0


def cmd_prompts_stats(args: argparse.Namespace) -> int:
    """Print prompt-store size, row counts and the dedup ratio.

    The dedup ratio is the headline number, not a curiosity: it is links per
    stored message -- how many times the fleet resent a message already stored
    -- so it measures the prompt-cache opportunity directly on real traffic,
    on a fleet whose agents resend whole conversations every turn.
    """
    from .prompt_store import open_db, stats

    console = Console()
    path = _prompt_db_path(args)
    if not path:
        console.print(
            "[red]No prompt database:[/red] pass --db PATH or set LLM_RELAY_PROMPT_DB"
        )
        return 1
    if not os.path.exists(path):
        console.print(f"No prompt store at {path}: nothing captured yet.")
        return 0

    conn = open_db(path)
    try:
        s = stats(conn, path)
    finally:
        conn.close()

    table = Table(title="Prompt store")
    table.add_column("Measure", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Requests", f"{s['requests']:,}")
    table.add_row("Stored messages", f"{s['stored_messages']:,}")
    table.add_row("Message links", f"{s['message_links']:,}")
    table.add_row("Dedup ratio", f"{s['dedup_ratio']:.2f}x")
    table.add_row("On disk", f"{s['bytes']:,} bytes")
    table.add_row("Content before compression", f"{s['uncompressed_bytes']:,} bytes")
    table.add_row("Distinct days", f"{s['distinct_days']:,}")
    table.add_row("Codec", str(s["codec"]))
    console.print(table)
    console.print(
        "Dedup ratio = links per stored message = the prompt-cache opportunity."
    )
    return 0


def cmd_prompts(args: argparse.Namespace) -> int:
    console = Console()
    action = getattr(args, "prompts_action", None)
    if action == "prune":
        return cmd_prompts_prune(args)
    if action == "stats":
        return cmd_prompts_stats(args)
    console.print("Usage: llm-relay prompts {prune,stats} --help")
    return 1


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

    p_cache = subparsers.add_parser(
        "cache-sample",
        help="Sample backend prefix-cache counters into the usage store",
    )
    p_cache.add_argument(
        "--usage-db", default=None,
        help="SQLite usage database (default: $LLM_RELAY_USAGE_DB)",
    )
    p_cache.add_argument(
        "--backend", action="append", default=[], metavar="URL",
        help="Backend root URL to scrape, repeatable. Default: every enabled "
             "provider/port in the loaded config (addresses are never hardcoded)",
    )
    p_cache.add_argument(
        "--day", default=None, help="Attribute deltas to this day (default: today UTC)"
    )
    p_cache.add_argument(
        "--timeout", type=float, default=5.0, help="Per-backend scrape timeout, seconds"
    )
    p_cache.add_argument(
        "--dry-run", action="store_true",
        help="Print what each backend reports without writing or moving cursors",
    )

    p_prompts = subparsers.add_parser(
        "prompts", help="Inspect and prune the captured prompt store"
    )
    prompts_sub = p_prompts.add_subparsers(dest="prompts_action")
    pp_prune = prompts_sub.add_parser(
        "prune",
        help="Delete prompt content older than a cutoff day (default: nothing, "
             "retention is indefinite)",
    )
    pp_prune.add_argument(
        "--db", default=None,
        help="SQLite prompt database (default: $LLM_RELAY_PROMPT_DB)",
    )
    pp_prune.add_argument(
        "--older-than", default=None, metavar="YYYY-MM-DD",
        help="Delete requests before this day. Without it, the cutoff comes from "
             "LLM_RELAY_PROMPT_RETENTION_DAYS; unset or 0 means keep forever",
    )
    pp_prune.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without writing",
    )
    pp_stats = prompts_sub.add_parser(
        "stats", help="Print store size, row counts and the dedup ratio"
    )
    pp_stats.add_argument(
        "--db", default=None,
        help="SQLite prompt database (default: $LLM_RELAY_PROMPT_DB)",
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
    if args.command == "cache-sample":
        return cmd_cache_sample(args)
    if args.command == "prompts":
        return cmd_prompts(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
