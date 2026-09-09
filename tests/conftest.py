"""Shared pytest fixtures for the bookchanger test suite.

Calls tests.yhm_stubs.install() BEFORE the module under test is imported,
so bookchanger.py's top-level imports always resolve even when fastapi /
pydub / speech_recognition / httpx are not installed on this machine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Make sure the stubs are in place before anything imports bookchanger.
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yhm_stubs  # noqa: E402  (local module, path set up above)
from fake_yhm import FakeYHM  # noqa: E402

STUB_STATUS = yhm_stubs.install()


def _load_bookchanger():
    """Import bookchanger.py by file path (it has no package structure)."""
    module_path = _TESTS_DIR.parent / "bookchanger.py"
    spec = importlib.util.spec_from_file_location("bookchanger_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so test modules can `import bookchanger_under_test`
    # and share this exact instance (module_from_spec alone does NOT register).
    sys.modules["bookchanger_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def bc():
    """The bookchanger module under test, with externals stubbed if needed."""
    return _load_bookchanger()


@pytest.fixture()
def yhm(bc, monkeypatch):
    """Wire every REST helper of bookchanger.py to a fresh FakeYHM."""
    fake = FakeYHM()
    monkeypatch.setattr(bc, "yhm_read_text_file", fake.read_text_file)
    monkeypatch.setattr(bc, "yhm_write_text_file", fake.write_text_file)
    monkeypatch.setattr(bc, "yhm_delete_files", fake.delete_files)
    return fake


@pytest.fixture()
def ini_path():
    """The canonical ivr2 ini path the endpoint builds from ext='100'."""
    return "ivr2:/100/ListAllInformation.ini"
