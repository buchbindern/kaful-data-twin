"""M2a: real tests for FilesystemObjectStore, using pytest's tmp_path fixture
(a fresh temp dir per test, auto-cleaned). This is what pytest buys us over a
plain script."""

import pytest

from storage import FilesystemObjectStore


def test_put_get_roundtrip(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    store.put("phm2010/c1/000001.npy.gz", b"\x1f\x8b hello")
    assert store.get("phm2010/c1/000001.npy.gz") == b"\x1f\x8b hello"


def test_exists(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    assert not store.exists("missing/key")
    store.put("a/b", b"x")
    assert store.exists("a/b")


def test_get_missing_raises_keyerror(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("nope")


def test_overwrite(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    store.put("k", b"first")
    store.put("k", b"second")
    assert store.get("k") == b"second"


def test_nested_dirs_created(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    store.put("deep/nested/path/blob.bin", b"data")
    assert (tmp_path / "deep" / "nested" / "path" / "blob.bin").read_bytes() == b"data"


def test_unsafe_key_rejected(tmp_path):
    store = FilesystemObjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("../escape", b"x")


def test_survives_reopen(tmp_path):
    # a new store instance over the same root sees prior data (persistence)
    FilesystemObjectStore(tmp_path).put("k", b"persisted")
    assert FilesystemObjectStore(tmp_path).get("k") == b"persisted"
