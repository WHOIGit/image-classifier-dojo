import pytest

from dojo.storage import LocalStorage, StorageError, get_storage


def test_write_read_roundtrip(tmp_path):
    storage = get_storage()
    assert isinstance(storage, LocalStorage)
    target = tmp_path / "nested" / "blob.bin"
    payload = b"\x89PNG\r\n\x00bytes"

    assert storage.exists(str(target)) is False
    storage.write_bytes(str(target), payload)  # creates parent dirs
    assert storage.exists(str(target)) is True
    assert storage.read_bytes(str(target)) == payload


def test_localize_returns_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = get_storage()
    local = storage.localize("./runs/x/cfg.yaml")
    assert local.is_absolute()
    assert local == (tmp_path / "runs" / "x" / "cfg.yaml")


def test_file_scheme_is_local(tmp_path):
    storage = get_storage()
    target = tmp_path / "blob.bin"
    storage.write_bytes(f"file://{target}", b"data")
    assert storage.read_bytes(str(target)) == b"data"


def test_missing_read_raises_file_not_found(tmp_path):
    storage = get_storage()
    with pytest.raises(FileNotFoundError):
        storage.read_bytes(str(tmp_path / "absent.bin"))


def test_non_local_scheme_is_a_clear_seam():
    storage = get_storage()
    with pytest.raises(StorageError, match="local-only"):
        storage.read_bytes("s3://bucket/key")
