"""
Database Service
Handles database initialization and connections
"""

import aiosqlite
from pathlib import Path


async def init_database(db_path: str):
    """Initialize database with schema and seed data"""
    # Read schema - look in project root
    # backend/app/services/database.py -> go up 3 levels to project root, then into database/
    project_root = Path(__file__).parent.parent.parent.parent
    schema_path = project_root / "database" / "schema.sql"
    
    if not schema_path.exists():
        print(f"⚠ Schema file not found at {schema_path}")
        return
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Create database
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(schema_sql)
        await db.commit()
        print("✓ Database schema created")
    
    # Always load seed data
    seed_path = project_root / "database" / "seed.sql"
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_sql = f.read()
        
        async with aiosqlite.connect(db_path) as db:
            # Check if data already exists
            cursor = await db.execute("SELECT COUNT(*) FROM patients")
            count = (await cursor.fetchone())[0]
            
            if count == 0:
                # Only insert seed data if database is empty
                await db.executescript(seed_sql)
                await db.commit()
                print("✓ Seed data loaded")
            else:
                print(f"✓ Database already has {count} patients, skipping seed data")
    else:
        print(f"⚠ Seed file not found at {seed_path}")


async def get_db_connection(db_path: str):
    """Get database connection"""
    return await aiosqlite.connect(db_path)
