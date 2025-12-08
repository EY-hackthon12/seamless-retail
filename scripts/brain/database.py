import psycopg2
import psycopg2.extras
import json
import os
import logging
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config - STRICT POSTGRES
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "seamless_retail")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"!! Critical: PostgreSQL Connection Failed: {e}")
        logger.error("Ensure PostgreSQL is running and credentials are correct.")
        return None

def init_db():
    conn = get_connection()
    if not conn:
        logger.warning("Skipping DB initialization due to connection failure.")
        return
        
    cur = conn.cursor()
    
    # User Profiles Table
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    created_at TIMESTAMP,
                    preferences JSONB
                )''')
    
    # Interactions/Memories Table for RAG
    cur.execute('''CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    text_content TEXT,
                    timestamp TIMESTAMP,
                    embedding_id INTEGER
                )''')
                
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"--> Database initialized (PostgreSQL at {DB_HOST}:{DB_PORT})")

def get_user(user_id):
    conn = get_connection()
    if not conn: return None
    
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
    row = cur.fetchone()
    
    result = None
    if row:
        result = {
            "user_id": row['user_id'],
            "name": row['name'],
            "email": row['email'],
            "created_at": row['created_at'].isoformat() if row['created_at'] else None,
            "preferences": row['preferences'] if row['preferences'] else {}
        }
    
    cur.close()
    conn.close()
    return result

def upsert_user(user_id, name, email, preferences=None):
    conn = get_connection()
    if not conn: return
    
    cur = conn.cursor()
    
    # Upsert using ON CONFLICT
    prefs_json = json.dumps(preferences if preferences else {})
    now = datetime.now()
    
    cur.execute('''
        INSERT INTO users (user_id, name, email, created_at, preferences)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            name = EXCLUDED.name,
            email = EXCLUDED.email,
            preferences = users.preferences || EXCLUDED.preferences
    ''', (user_id, name, email, now, prefs_json))
                  
    conn.commit()
    cur.close()
    conn.close()

def add_interaction(user_id, text):
    conn = get_connection()
    if not conn: return None
    
    cur = conn.cursor()
    now = datetime.now()
    # Postgres specific RETURNING syntax
    cur.execute("INSERT INTO interactions (user_id, text_content, timestamp) VALUES (%s, %s, %s) RETURNING interaction_id",
              (user_id, text, now))
    interaction_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return interaction_id

def get_user_interactions(user_id, limit=50):
    conn = get_connection()
    if not conn: return []
    
    cur = conn.cursor()
    cur.execute("SELECT text_content, timestamp FROM interactions WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"text": r[0], "timestamp": r[1].isoformat()} for r in rows]

if __name__ == "__main__":
    init_db()
