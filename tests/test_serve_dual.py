"""Dual-listener socket builder: fail-closed on an empty key store."""
from __future__ import annotations

import socket

from llm_relay.api.app import build_sockets
from llm_relay.auth import AuthConfig, Principal


def _free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def test_build_sockets_dual_when_keys_exist():
    p1, p2 = _free_ports(2)
    cfg = AuthConfig(enabled=True, principals_by_hash={"h": Principal(id="x")})
    socks, warnings = build_sockets("127.0.0.1", p1, p2, cfg)
    assert len(socks) == 2 and not warnings
    for s in socks:
        s.close()


def test_build_sockets_refuses_auth_port_with_empty_store():
    p1, p2 = _free_ports(2)
    cfg = AuthConfig(enabled=True, principals_by_hash={})
    socks, warnings = build_sockets("127.0.0.1", p1, p2, cfg)
    assert len(socks) == 1 and warnings
    socks[0].close()


def test_build_sockets_refuses_auth_port_when_all_keys_disabled():
    p1, p2 = _free_ports(2)
    cfg = AuthConfig(
        enabled=True,
        principals_by_hash={"h": Principal(id="x", enabled=False)},
    )
    socks, warnings = build_sockets("127.0.0.1", p1, p2, cfg)
    assert len(socks) == 1 and warnings
    socks[0].close()


def test_build_sockets_single_when_no_auth_port():
    (p1,) = _free_ports(1)
    socks, warnings = build_sockets("127.0.0.1", p1, None, AuthConfig())
    assert len(socks) == 1 and not warnings
    socks[0].close()
