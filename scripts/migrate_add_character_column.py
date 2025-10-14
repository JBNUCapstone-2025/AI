"""
Migration script to add/modify character column in User table to be nullable
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.database import engine
from dotenv import load_dotenv

load_dotenv()


def migrate():
    """Add or modify character column to be nullable"""

    with engine.connect() as connection:
        try:
            # Check if column exists
            result = connection.execute(text("""
                SELECT COLUMN_NAME, IS_NULLABLE, COLUMN_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'User'
                AND COLUMN_NAME = 'character'
            """))

            column_info = result.fetchone()

            if column_info:
                print(f"✓ Column 'character' exists")
                print(f"  - Type: {column_info[2]}")
                print(f"  - Nullable: {column_info[1]}")

                if column_info[1] == 'NO':
                    print("\n→ Modifying column to be nullable...")
                    connection.execute(text("""
                        ALTER TABLE `User`
                        MODIFY COLUMN `character` VARCHAR(50) NULL
                    """))
                    connection.commit()
                    print("✓ Column modified successfully!")
                else:
                    print("✓ Column is already nullable. No changes needed.")
            else:
                print("→ Column 'character' does not exist. Adding it...")
                connection.execute(text("""
                    ALTER TABLE `User`
                    ADD COLUMN `character` VARCHAR(50) NULL
                """))
                connection.commit()
                print("✓ Column added successfully!")

        except Exception as e:
            print(f"✗ Error during migration: {e}")
            connection.rollback()
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add/Modify character column")
    print("=" * 60)
    print()

    migrate()

    print()
    print("=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
