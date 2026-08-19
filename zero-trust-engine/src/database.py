"""SQLite Database Management for Zero Trust Engine."""
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite connections, migrations, and schema initialization."""
    
    def __init__(self, db_path: str | Path = "data/policy_log.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize_schema()
        
    @property
    def connection(self) -> sqlite3.Connection:
        """Thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def initialize_schema(self):
        """Create tables and indexes if they don't exist."""
        try:
            with self.connection as conn:
                cursor = conn.cursor()
                
                # Schema Info for Migrations
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_info (
                        version TEXT PRIMARY KEY,
                        applied_on TEXT NOT NULL
                    )
                ''')
                
                # Check current version
                cursor.execute('SELECT version FROM schema_info ORDER BY applied_on DESC LIMIT 1')
                row = cursor.fetchone()
                if not row:
                    # Apply V1 migration
                    self._apply_v1_schema(cursor)
                    from datetime import timezone
                    cursor.execute(
                        'INSERT INTO schema_info (version, applied_on) VALUES (?, ?)',
                        ('1.0.0', datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
                    )
                    logger.info("Database schema initialized to version 1.0.0")
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    def _apply_v1_schema(self, cursor: sqlite3.Cursor):
        """Apply the initial V1 schema."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policy_log (
                audit_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                trust_score REAL NOT NULL,
                trust_threshold REAL NOT NULL,
                trust_state TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                policy_version TEXT NOT NULL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device ON policy_log(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON policy_log(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_decision ON policy_log(decision)')

    def execute_transaction(self, query: str, params: tuple):
        """Execute a write query safely within a transaction."""
        try:
            with self.connection as conn:
                conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"Database transaction failed: {e}")
            raise

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Fetch multiple rows as dictionaries."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Database fetch failed: {e}")
            raise
            
    def get_decision_stats(self) -> dict:
        """Get aggregate statistics for the health endpoint."""
        stats = {
            "total_decisions": 0,
            "allow": 0,
            "verify": 0,
            "block": 0
        }
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT decision, COUNT(*) as count FROM policy_log GROUP BY decision")
            for row in cursor.fetchall():
                decision = row['decision'].lower()
                count = row['count']
                stats[decision] = count
                stats["total_decisions"] += count
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch decision stats: {e}")
        return stats
