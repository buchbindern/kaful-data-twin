"""Phase E: rate limiting, security headers, session lifecycle, input limits."""
import time
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from auth import RateLimiter, hash_password, new_session_token, session_expiry
from domain.models import Machine, Run, Cut, FeatureRecord, WearLabel, User, Session
from storage import SQLiteDataStore
from twin import build_twin
from api import create_app
import numpy as np

RNG = np.random.default_rng(0)


def _seed(store_dir):
    ds = SQLiteDataStore(store_dir / "kaful.db")
    ds.create_machine(Machine("phm2010", "phm2010_milling"))
    ds.create_run(Run("c1", "phm2010"))
    for cut in range(1, 60):
        w = 0.05 + 0.12 * (cut / 60) ** 2.5
        ds.append_cut(Cut("c1", cut, f"k{cut}"))
        ds.append_features(FeatureRecord("c1", cut, {"vibration_x_mean_abs": 1521 * w ** 2}))
        ds.append_wear_label(WearLabel("c1", cut, w))
    ds.save_twin_state(build_twin(ds, "c1", n_particles=400)); ds.close()


@pytest.fixture
def client(tmp_path):
    _seed(tmp_path)
    return TestClient(create_app(store_dir=str(tmp_path)))


# ---- RateLimiter unit ----
def test_ratelimiter_blocks_over_limit():
    rl = RateLimiter(limit=3, window_seconds=60)
    assert all(rl.allow("ip") for _ in range(3))
    assert not rl.allow("ip")            # 4th blocked
    assert rl.allow("other")             # different key unaffected

def test_ratelimiter_window_expiry():
    rl = RateLimiter(limit=1, window_seconds=0.2)
    assert rl.allow("k") and not rl.allow("k")
    time.sleep(0.25)
    assert rl.allow("k")                 # window slid


# ---- endpoint rate limiting ----
def test_login_rate_limited(client, monkeypatch):
    monkeypatch.setenv("KAFUL_LOGIN_RATE", "5")   # note: app already built; test default 10
    # default login limit is 10/min; 11th should 429
    codes = [client.post("/auth/login", json={"email": "x@y.com", "password": "nope12345"}).status_code
             for _ in range(11)]
    assert 429 in codes
    assert codes.count(401) >= 1                   # earlier ones were normal auth failures

def test_signup_rate_limited(client):
    codes = [client.post("/auth/signup", json={"email": f"u{i}@y.com", "password": "password123"}).status_code
             for i in range(7)]
    assert 429 in codes                            # signup limit is 5/min


# ---- security headers ----
def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Referrer-Policy" in r.headers


# ---- input limits ----
def test_overlong_password_is_400(client):
    r = client.post("/auth/signup", json={"email": "long@y.com", "password": "x" * 200})
    assert r.status_code == 400

def test_overlong_email_is_400(client):
    r = client.post("/auth/signup", json={"email": "a" * 250 + "@y.com", "password": "password123"})
    assert r.status_code == 400


# ---- session lifecycle ----
def test_logout_all_revokes_every_session(tmp_path):
    _seed(tmp_path)
    app = create_app(store_dir=str(tmp_path))
    a = TestClient(app); b = TestClient(app)
    a.post("/auth/signup", json={"email": "z@y.com", "password": "password123"})
    b.post("/auth/login", json={"email": "z@y.com", "password": "password123"})  # 2nd session, same user
    assert a.get("/auth/me").status_code == 200 and b.get("/auth/me").status_code == 200
    a.post("/auth/logout-all")
    assert a.get("/auth/me").status_code == 401 and b.get("/auth/me").status_code == 401  # both gone

def test_expired_session_cleanup(tmp_path):
    _seed(tmp_path)
    ds = SQLiteDataStore(tmp_path / "kaful.db")
    ds.create_user(User("u", "c@y.com", hash_password("password123"), datetime.now(timezone.utc)))
    past = datetime.now(timezone.utc) - timedelta(days=1)
    ds.create_session(Session("expiredtok", "u", past, past))
    ds.create_session(Session("livetok", "u", datetime.now(timezone.utc), session_expiry()))
    ds.delete_expired_sessions(datetime.now(timezone.utc))
    assert ds.get_valid_session("livetok") is not None      # live kept
    row = ds._conn.execute("SELECT count(*) FROM sessions WHERE token='expiredtok'").fetchone()[0]
    assert row == 0                                          # expired row physically removed
    ds.close()
