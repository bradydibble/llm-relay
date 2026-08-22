"""The stores hold coworkers' token history and conversations. The files must
not be created world-readable.

Nothing in the shipped units sets UMask, so the service default (0022) yields
0644 databases. Production is currently saved only by a hand-set 0750 on the
parent directory -- not by anything in this repo, and LLM_RELAY_USAGE_DB is
explicitly designed to be relocatable. auth.py already chmods its key-hash file
to 0600; the conversation archive deserves at least the same.
"""
from __future__ import annotations

import os
import stat

import pytest

from llm_relay import prompt_store, usage_store

WORLD_OR_GROUP = stat.S_IRWXG | stat.S_IRWXO


@pytest.mark.parametrize("mod,name", [(usage_store, "usage.db"),
                                      (prompt_store, "prompts.db")])
def test_database_file_is_not_group_or_world_readable(tmp_path, mod, name):
    old = os.umask(0o022)  # the service default, which is the real-world case
    try:
        path = tmp_path / "state" / name
        conn = mod.open_db(str(path))
        conn.close()
    finally:
        os.umask(old)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert not (mode & WORLD_OR_GROUP), f"{name} is {oct(mode)}, readable beyond the owner"


@pytest.mark.parametrize("mod,name", [(usage_store, "usage.db"),
                                      (prompt_store, "prompts.db")])
def test_created_parent_directory_is_not_group_or_world_readable(tmp_path, mod, name):
    old = os.umask(0o022)
    try:
        parent = tmp_path / "fresh"
        conn = mod.open_db(str(parent / name))
        conn.close()
    finally:
        os.umask(old)
    mode = stat.S_IMODE(os.stat(parent).st_mode)
    assert not (mode & WORLD_OR_GROUP), f"parent dir is {oct(mode)}"


@pytest.mark.parametrize("mod,name", [(usage_store, "usage.db"),
                                      (prompt_store, "prompts.db")])
def test_reopening_does_not_loosen_an_existing_file(tmp_path, mod, name):
    path = tmp_path / name
    mod.open_db(str(path)).close()
    os.chmod(path, 0o600)
    mod.open_db(str(path)).close()
    assert not (stat.S_IMODE(os.stat(path).st_mode) & WORLD_OR_GROUP)
