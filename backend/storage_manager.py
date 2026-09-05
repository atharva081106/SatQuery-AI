import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "queries.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                query_text TEXT,
                timestamp TEXT,
                response_json TEXT,
                geojson_data TEXT,
                image_paths TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

def save_query_result(
    query_text: str,
    response: Dict[str, Any],
    query_id: Optional[str] = None,
    geojson_data: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None
) -> str:
    """
    Persists an analysis execution into SQLite queries.db.
    Returns the unique query_id.
    """
    init_db()
    if not query_id:
        query_id = str(uuid.uuid4())
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    geojson_str = json.dumps(geojson_data) if geojson_data else ""
    image_paths_str = json.dumps(image_paths) if image_paths else ""
    response_str = json.dumps(response)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO queries (id, query_text, timestamp, response_json, geojson_data, image_paths)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query_id, query_text, timestamp, response_str, geojson_str, image_paths_str))
        conn.commit()
    finally:
        conn.close()
        
    return query_id

def get_query_result(query_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a previously computed query result by its unique ID.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, query_text, timestamp, response_json, geojson_data FROM queries WHERE id = ?", (query_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        result_id, q_text, ts, resp_json, geojson_str = row
        data = json.loads(resp_json) if resp_json else {}
        data["id"] = result_id
        data["saved_timestamp"] = ts
        if geojson_str and "geojson_data" not in data:
            try:
                data["geojson_data"] = json.loads(geojson_str)
            except Exception:
                pass
        return data
    finally:
        conn.close()

def list_recent_queries(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Returns recent queries for telemetry, history, and audit trails.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, query_text, timestamp, response_json 
            FROM queries 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            qid, qtext, ts, resp_json = row
            summary = {
                "id": qid,
                "query": qtext,
                "timestamp": ts,
                "status": "success",
                "task": "UNKNOWN",
                "confidence": 0.0,
                "has_geojson": False
            }
            if resp_json:
                try:
                    parsed = json.loads(resp_json)
                    exec_sum = parsed.get("execution_summary", {})
                    summary["task"] = exec_sum.get("selected_task", parsed.get("task", "ANALYSIS"))
                    summary["confidence"] = parsed.get("confidence", 0.0)
                    summary["spatial_coherence"] = parsed.get("spatial_coherence_score", 1.0)
                    summary["has_geojson"] = bool(parsed.get("geojson_data"))
                except Exception:
                    pass
            results.append(summary)
        return results
    finally:
        conn.close()
