"""
Tests for Authentication, Password Hashing, User Management, and Role Authorization.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from database.db import init_db
from database.repository import (
    count_users,
    create_user,
    delete_user,
    get_user_by_id,
    get_user_by_username,
    hash_password,
    list_all_users,
    update_user_last_login,
    update_user_status,
    verify_password,
)


def test_password_hashing_and_verification():
    """Test PBKDF2 password hashing and verification logic."""
    raw_password = "SecretPassword123!"
    pwd_hash = hash_password(raw_password)

    # Must not store plaintext
    assert raw_password not in pwd_hash
    assert "$" in pwd_hash

    # Must verify correct password
    assert verify_password(pwd_hash, raw_password) is True

    # Must reject incorrect password
    assert verify_password(pwd_hash, "WrongPassword!") is False
    assert verify_password(pwd_hash, "") is False
    assert verify_password("", raw_password) is False


def test_user_repository_crud(tmp_path: Path):
    """Test full CRUD operations on user accounts."""
    db_file = tmp_path / "test_auth.db"
    init_db(db_file)

    assert count_users(db_file) == 0

    # 1. Create User
    pwd_hash = hash_password("investigator_pass")
    u_id = create_user(db_file, "detective_smith", pwd_hash, role="INVESTIGATOR", is_active=True)
    assert u_id > 0
    assert count_users(db_file) == 1

    # 2. Get User
    user = get_user_by_username(db_file, "detective_smith")
    assert user is not None
    assert user["id"] == u_id
    assert user["username"] == "detective_smith"
    assert user["role"] == "INVESTIGATOR"
    assert user["is_active"] == 1

    # 3. Update status & last login
    update_user_status(db_file, u_id, is_active=False)
    updated_user = get_user_by_id(db_file, u_id)
    assert updated_user["is_active"] == 0

    update_user_last_login(db_file, u_id)
    login_user = get_user_by_id(db_file, u_id)
    assert login_user["last_login"] is not None

    # 4. List users
    all_users = list_all_users(db_file)
    assert len(all_users) == 1

    # 5. Delete user
    delete_user(db_file, u_id)
    assert count_users(db_file) == 0
