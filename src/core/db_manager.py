import aiosqlite
import asyncio
import os
import json

DB_DIR = "data/dbs"

class AsyncDBManager:
    def __init__(self, model_name: str = "router"):
        self.model_name = model_name

        # Ensure dir exists
        os.makedirs(DB_DIR, exist_ok=True)

        # 1. Shared DB for Global Engrams & DPO Data
        self.shared_db_path = os.path.join(DB_DIR, "shared_engram.sqlite")
        # 2. Model-Specific DB for Local Context and isolated reasoning
        self.model_db_path = os.path.join(DB_DIR, f"{model_name}_memory.sqlite")

    async def setup(self):
        """Initialize the non-blocking databases asynchronously."""
        # Setup Shared Database
        async with aiosqlite.connect(self.shared_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dpo_dataset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    prompt TEXT,
                    chosen TEXT,
                    rejected TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS global_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

        # Setup Model-Specific Database
        async with aiosqlite.connect(self.model_db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    # --- DIRECT PREFERENCE OPTIMIZATION (DPO) PIPELINE ---

    async def record_dpo_preference(self, prompt: str, chosen: str, rejected: str):
        """
        Logs a winning and losing response for future Direct Preference Optimization.
        This allows ElfAgent to build a local dataset for continuous alignment.
        """
        async with aiosqlite.connect(self.shared_db_path) as db:
            await db.execute(
                "INSERT INTO dpo_dataset (model_name, prompt, chosen, rejected) VALUES (?, ?, ?, ?)",
                (self.model_name, prompt, chosen, rejected)
            )
            await db.commit()

    # --- SHARED MEMORY (ENGRAMS) ---

    async def set_shared_memory(self, key: str, value: dict):
        async with aiosqlite.connect(self.shared_db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO global_memory (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )
            await db.commit()

    async def get_shared_memory(self, key: str):
        async with aiosqlite.connect(self.shared_db_path) as db:
            async with db.execute("SELECT value FROM global_memory WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else None

    # --- ISOLATED MODEL CONTEXT ---

    async def log_model_interaction(self, session_id: str, role: str, content: str):
        async with aiosqlite.connect(self.model_db_path) as db:
            await db.execute(
                "INSERT INTO model_context (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            await db.commit()
