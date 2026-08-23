import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / "database.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            timestamp TEXT,
            inference_time_seconds REAL,
            findings TEXT,
            impression TEXT,
            status TEXT,
            error TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_history(image_name: str, inference_time_seconds: float, findings: str, impression: str, status: str, error: str = ""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO inference_history (image_name, timestamp, inference_time_seconds, findings, impression, status, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (image_name, timestamp, inference_time_seconds, findings, impression, status, error))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_history() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM inference_history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_history_item(item_id: int) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM inference_history WHERE id = ?', (item_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
