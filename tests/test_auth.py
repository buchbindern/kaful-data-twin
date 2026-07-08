import uuid
from datetime import datetime, timezone, timedelta

import pytest

from auth import hash_password, verify_password, new_session_token, session_expiry
from domain.models import User, Session
from storage import SQLiteDataStore


def test_password_hash_roundtrip():
    h = hash_password("correct horse battery")
    assert h != "correct horse battery"          # never plaintext
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong password", h)


def test_password_rejects_overlong():
    with pytest.raises(ValueError):
        hash_password("x" * 200)


def test_session_token_is_unique_and_long():
    a, b = new_session_token(), new_session_token()
    assert a != b and len(a) >= 32


def _store(tmp_path):
    return SQLiteDataStore(tmp_path / "kaful.db")


def _user(email="a@b.com"):
    return User(str(uuid.uuid4()), email, hash_password("pw-123456"), datetime.now(timezone.utc))


def test_user_persistence(tmp_path):
    ds = _store(tmp_path); u = _user()
    ds.create_user(u)
    assert ds.get_user_by_email("a@b.com").user_id == u.user_id
    assert ds.get_user(u.user_id).email == "a@b.com"
    assert ds.get_user_by_email("missing@b.com") is None
    ds.close()


def test_duplicate_email_rejected(tmp_path):
    ds = _store(tmp_path); ds.create_user(_user())
    with pytest.raises(Exception):
        ds.create_user(_user())          # same email -> UNIQUE violation
    ds.close()


def test_session_lifecycle(tmp_path):
    ds = _store(tmp_path); u = _user(); ds.create_user(u)
    tok = new_session_token()
    ds.create_session(Session(tok, u.user_id, datetime.now(timezone.utc), session_expiry()))
    assert ds.get_valid_session(tok).user_id == u.user_id
    ds.delete_session(tok)
    assert ds.get_valid_session(tok) is None


def test_expired_session_is_invalid(tmp_path):
    ds = _store(tmp_path); u = _user(); ds.create_user(u)
    tok = new_session_token()
    past = datetime.now(timezone.utc) - timedelta(days=1)
    ds.create_session(Session(tok, u.user_id, past, past))   # already expired
    assert ds.get_valid_session(tok) is None
    ds.close()


# ---------------- endpoint tests ----------------
import pytest
from fastapi.testclient import TestClient
from api import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(store_dir=str(tmp_path)))


def test_signup_login_me_logout_flow(client):
    # signup sets a session cookie
    r = client.post("/auth/signup", json={"email": "Alice@Example.com ", "password": "hunter2222"})
    assert r.status_code == 200 and r.json()["email"] == "alice@example.com"   # normalized
    # authenticated
    me = client.get("/auth/me")
    assert me.status_code == 200 and me.json()["email"] == "alice@example.com"
    # logout clears the session
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").status_code == 401
    # log back in
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": "hunter2222"}).status_code == 200
    assert client.get("/auth/me").status_code == 200


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_duplicate_email_is_409(client):
    client.post("/auth/signup", json={"email": "b@c.com", "password": "password1"})
    r = client.post("/auth/signup", json={"email": "b@c.com", "password": "password2"})
    assert r.status_code == 409


def test_bad_email_and_short_password_rejected(client):
    assert client.post("/auth/signup", json={"email": "not-an-email", "password": "password1"}).status_code == 400
    assert client.post("/auth/signup", json={"email": "ok@x.com", "password": "short"}).status_code == 400


def test_wrong_password_is_401_generic(client):
    client.post("/auth/signup", json={"email": "d@e.com", "password": "password1"})
    r = client.post("/auth/login", json={"email": "d@e.com", "password": "WRONG"})
    assert r.status_code == 401 and "invalid email or password" in r.json()["detail"]  # no enumeration


def test_unknown_user_login_is_same_401(client):
    r = client.post("/auth/login", json={"email": "nobody@x.com", "password": "password1"})
    assert r.status_code == 401 and "invalid email or password" in r.json()["detail"]
