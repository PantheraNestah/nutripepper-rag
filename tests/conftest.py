# Point the app at a throwaway SQLite file before any app module is imported,
# so tests never touch the real nutripepper.db.
import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='nutripepper_test_')}/test.db"
