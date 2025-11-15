"""
User Authentication Module
Handles user registration, login, and credential management
"""

import os
import json
import hashlib


class UserAuth:
    """Handle user authentication operations"""

    def __init__(self, users_file="data/users.json"):
        """
        Initialize UserAuth with file path for storing user credentials

        Args:
            users_file (str): Path to the JSON file storing user data
        """
        self.users_file = users_file
        self._ensure_data_directory()
        self._ensure_users_file()

    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        directory = os.path.dirname(self.users_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _ensure_users_file(self):
        """Create users file if it doesn't exist"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, "w") as f:
                json.dump({}, f)

    def _hash_password(self, password):
        """
        Hash password using SHA-256

        Args:
            password (str): Plain text password

        Returns:
            str: Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def _load_users(self):
        """
        Load users from file

        Returns:
            dict: Dictionary of users and their credentials
        """
        try:
            with open(self.users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_users(self, users):
        """
        Save users to file

        Args:
            users (dict): Dictionary of users to save
        """
        with open(self.users_file, "w") as f:
            json.dump(users, f, indent=4)

    def register(self, username, password):
        """
        Register a new user

        Args:
            username (str): Username for new account
            password (str): Password for new account

        Returns:
            tuple: (success: bool, message: str)
        """
        # Validate inputs
        if not username or not password:
            return False, "Username and password cannot be empty"

        if len(username) < 3:
            return False, "Username must be at least 3 characters long"

        if len(password) < 6:
            return False, "Password must be at least 6 characters long"

        # Check if username contains only alphanumeric and underscore
        if not username.replace("_", "").isalnum():
            return False, "Username can only contain letters, numbers, and underscores"

        users = self._load_users()

        # Check if user already exists
        if username in users:
            return False, "Username already exists"

        # Save new user
        users[username] = {
            "password": self._hash_password(password),
            "created_at": self._get_timestamp(),
        }
        self._save_users(users)

        return True, "Registration successful!"

    def login(self, username, password):
        """
        Authenticate user login

        Args:
            username (str): Username
            password (str): Password

        Returns:
            tuple: (success: bool, message: str)
        """
        if not username or not password:
            return False, "Username and password cannot be empty"

        users = self._load_users()

        # Check if user exists
        if username not in users:
            return False, "Invalid username or password"

        # Verify password
        if users[username]["password"] == self._hash_password(password):
            return True, "Login successful!"
        else:
            return False, "Invalid username or password"

    def _get_timestamp(self):
        """
        Get current timestamp

        Returns:
            str: Current timestamp in ISO format
        """
        from datetime import datetime

        return datetime.now().isoformat()

    def user_exists(self, username):
        """
        Check if a user exists

        Args:
            username (str): Username to check

        Returns:
            bool: True if user exists, False otherwise
        """
        users = self._load_users()
        return username in users

    def get_all_usernames(self):
        """
        Get list of all registered usernames

        Returns:
            list: List of usernames
        """
        users = self._load_users()
        return list(users.keys())
