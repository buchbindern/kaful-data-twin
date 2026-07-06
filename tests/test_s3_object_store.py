"""S3ObjectStore against a mocked S3 (moto) + env-driven store selection.
Same contract as FilesystemObjectStore, so ingest/twin/API need no changes."""

import boto3
import pytest
from moto import mock_aws

from storage import S3ObjectStore, FilesystemObjectStore, object_store_from_env


@mock_aws
def test_put_get_exists_roundtrip():
    c = boto3.client("s3", region_name="us-east-1")
    c.create_bucket(Bucket="kaful-test")
    st = S3ObjectStore("kaful-test", prefix="kaful", client=c)
    st.put("phm2010/c1/000001.npy.gz", b"\x00rawbytes")
    assert st.exists("phm2010/c1/000001.npy.gz")
    assert st.get("phm2010/c1/000001.npy.gz") == b"\x00rawbytes"

@mock_aws
def test_missing_key_raises_keyerror():
    c = boto3.client("s3", region_name="us-east-1")
    c.create_bucket(Bucket="kaful-test")
    st = S3ObjectStore("kaful-test", client=c)
    assert not st.exists("nope")
    with pytest.raises(KeyError):
        st.get("nope")

@mock_aws
def test_prefix_is_applied():
    c = boto3.client("s3", region_name="us-east-1")
    c.create_bucket(Bucket="kaful-test")
    S3ObjectStore("kaful-test", prefix="kaful", client=c).put("a/b.gz", b"x")
    keys = [o["Key"] for o in c.list_objects_v2(Bucket="kaful-test")["Contents"]]
    assert keys == ["kaful/a/b.gz"]

def test_unsafe_key_rejected():
    st = S3ObjectStore("b", client=object())
    with pytest.raises(ValueError):
        st.put("/abs", b"x")
    with pytest.raises(ValueError):
        st.put("../escape", b"x")

def test_env_factory_defaults_to_filesystem(tmp_path, monkeypatch):
    monkeypatch.delenv("KAFUL_OBJECT_STORE", raising=False)
    assert isinstance(object_store_from_env(tmp_path), FilesystemObjectStore)

def test_env_factory_selects_s3(monkeypatch):
    monkeypatch.setenv("KAFUL_OBJECT_STORE", "s3")
    monkeypatch.setenv("KAFUL_S3_BUCKET", "kaful-test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x"); monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    store = object_store_from_env("unused")
    assert isinstance(store, S3ObjectStore) and store.bucket == "kaful-test"

def test_env_factory_s3_requires_bucket(monkeypatch):
    monkeypatch.setenv("KAFUL_OBJECT_STORE", "s3")
    monkeypatch.delenv("KAFUL_S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError):
        object_store_from_env("unused")
