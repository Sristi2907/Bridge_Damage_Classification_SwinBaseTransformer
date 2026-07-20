import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
from app.services.db_service import db_service
from app.core.config import settings

class TestEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure database is initialized
        db_service.initialize_database()

    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify nested structure
        self.assertIn("status", data)
        self.assertIn("database", data)
        self.assertIn("storage", data)
        self.assertIn("model", data)
        
        self.assertEqual(data["database"]["status"], "connected")
        self.assertEqual(data["database"]["path"], settings.SQLITE_DB_PATH)
        self.assertEqual(data["storage"]["status"], "available")
        self.assertEqual(data["model"]["status"], "loaded")
        self.assertEqual(data["model"]["architecture"], "Swin-B")

    def test_dashboard_stats(self):
        client = TestClient(app)
        response = client.get("/api/dashboard-stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify stats keys
        self.assertIn("total_analyses", data)
        self.assertIn("total_single_analyses", data)
        self.assertIn("total_batch_analyses", data)
        self.assertIn("average_processing_time", data)
        self.assertIn("average_confidence", data)
        self.assertIn("recent_analyses", data)

    def test_database_backup(self):
        client = TestClient(app)
        response = client.get("/api/database-backup")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/x-sqlite3"))
        
        # Check if backups folder contains the backup
        backup_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "backups")
        self.assertTrue(os.path.exists(backup_dir))
        files = os.listdir(backup_dir)
        self.assertTrue(len(files) > 0)
        self.assertTrue(any(f.startswith("backup_") and f.endswith(".db") for f in files))

if __name__ == "__main__":
    unittest.main()
