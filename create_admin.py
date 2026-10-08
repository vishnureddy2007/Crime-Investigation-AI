"""
Administrative CLI Utility to create or update an Admin user for Crime Investigation AI.

Usage:
    python create_admin.py [--username USERNAME] [--password PASSWORD] [--role ADMIN|INVESTIGATOR|VIEWER]
"""

import sys
import getpass
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH
from database.db import init_db
from database.repository import (
    hash_password,
    get_user_by_username,
    create_user,
    get_connection,
    _now_iso,
)


def main():
    parser = argparse.ArgumentParser(description="Create or update an admin user for Crime Investigation AI.")
    parser.add_argument("--username", help="Admin username")
    parser.add_argument("--password", help="Admin password")
    parser.add_argument("--role", default="ADMIN", choices=["ADMIN", "INVESTIGATOR", "VIEWER"], help="User role (default: ADMIN)")
    args = parser.parse_args()

    init_db(DATABASE_PATH)

    username = args.username
    password = args.password

    if not username:
        username = input("Enter Username: ").strip()

    if not username:
        print("[ERROR] Username cannot be empty.")
        sys.exit(1)

    if not password:
        password = getpass.getpass("Enter Password: ").strip()
        confirm = getpass.getpass("Confirm Password: ").strip()
        if password != confirm:
            print("[ERROR] Passwords do not match!")
            sys.exit(1)

    if not password or len(password) < 4:
        print("[ERROR] Password must be at least 4 characters long.")
        sys.exit(1)

    pwd_hash = hash_password(password)

    existing = get_user_by_username(DATABASE_PATH, username)
    if existing:
        with get_connection(DATABASE_PATH) as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, role = ?, is_active = 1 WHERE username = ?",
                (pwd_hash, args.role.upper(), username)
            )
        print(f"[OK] Successfully updated user '{username}' with role '{args.role.upper()}'.")
    else:
        uid = create_user(DATABASE_PATH, username, pwd_hash, role=args.role.upper(), is_active=True)
        print(f"[OK] Successfully created new user '{username}' (ID: {uid}) with role '{args.role.upper()}'.")


if __name__ == "__main__":
    main()
