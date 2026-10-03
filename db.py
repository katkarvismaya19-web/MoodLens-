"""SQLite database: users (salted PBKDF2 hashes) and emotion detections."""
import hashlib
import hmac
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, name TEXT NOT NULL,
            salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL,
            created TEXT NOT NULL, source TEXT NOT NULL, session_id TEXT,
            emotion TEXT NOT NULL, confidence REAL NOT NULL, image_path TEXT)""")


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def create_user(username, name, password):
    salt = os.urandom(16)
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                      (username.lower(), name, salt.hex(), _hash(password, salt),
                       datetime.now().isoformat(timespec="seconds")))
        return True
    except sqlite3.IntegrityError:
        return False


def verify(username, password):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.lower(),)).fetchone()
    if row and hmac.compare_digest(row["pw_hash"], _hash(password, bytes.fromhex(row["salt"]))):
        return {"username": row["username"], "name": row["name"]}
    return None


def add_detection(username, source, emotion, confidence, session_id=None, image_path=None):
    with _conn() as c:
        c.execute("INSERT INTO detections (username, created, source, session_id, emotion, "
                  "confidence, image_path) VALUES (?,?,?,?,?,?,?)",
                  (username, datetime.now().isoformat(timespec="seconds"), source,
                   session_id, emotion, confidence, image_path))


def detections(username):
    with _conn() as c:
        rows = c.execute("SELECT * FROM detections WHERE username=? ORDER BY id DESC",
                         (username,)).fetchall()
    return [dict(r) for r in rows]


def delete_detection(det_id, username):
    with _conn() as c:
        c.execute("DELETE FROM detections WHERE id=? AND username=?", (det_id, username))


def clear(username):
    with _conn() as c:
        c.execute("DELETE FROM detections WHERE username=?", (username,))
