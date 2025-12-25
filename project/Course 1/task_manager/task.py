"""
Task Management Module
Handles CRUD operations for user tasks
"""

import os
import json
from datetime import datetime


class TaskManager:
    """Manage tasks for authenticated users"""

    def __init__(self, username, tasks_dir="data/tasks"):
        """
        Initialize TaskManager for a specific user

        Args:
            username (str): Username of the authenticated user
            tasks_dir (str): Directory to store task files
        """
        self.username = username
        self.tasks_dir = tasks_dir
        self.tasks_file = os.path.join(tasks_dir, f"{username}_tasks.json")
        self._ensure_tasks_directory()
        self._ensure_tasks_file()

    def _ensure_tasks_directory(self):
        """Create tasks directory if it doesn't exist"""
        if not os.path.exists(self.tasks_dir):
            os.makedirs(self.tasks_dir)

    def _ensure_tasks_file(self):
        """Create tasks file for user if it doesn't exist"""
        if not os.path.exists(self.tasks_file):
            with open(self.tasks_file, "w") as f:
                json.dump([], f)

    def _load_tasks(self):
        """
        Load tasks from file

        Returns:
            list: List of task dictionaries
        """
        try:
            with open(self.tasks_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_tasks(self, tasks):
        """
        Save tasks to file

        Args:
            tasks (list): List of task dictionaries to save
        """
        with open(self.tasks_file, "w") as f:
            json.dump(tasks, f, indent=4)

    def _get_timestamp(self):
        """
        Get current timestamp

        Returns:
            str: Current timestamp in readable format
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_next_id(self, tasks):
        """
        Get next available task ID

        Args:
            tasks (list): Current list of tasks

        Returns:
            int: Next available ID
        """
        if not tasks:
            return 1
        return max(task["id"] for task in tasks) + 1

    def add_task(self, title, description="", priority="Medium", due_date=""):
        """
        Add a new task

        Args:
            title (str): Task title
            description (str): Task description (optional)
            priority (str): Task priority (Low/Medium/High)
            due_date (str): Due date for task (optional)

        Returns:
            tuple: (success: bool, message: str, task_id: int or None)
        """
        if not title or not title.strip():
            return False, "Task title cannot be empty", None

        if priority not in ["Low", "Medium", "High"]:
            priority = "Medium"

        tasks = self._load_tasks()
        task_id = self._get_next_id(tasks)

        new_task = {
            "id": task_id,
            "title": title.strip(),
            "description": description.strip(),
            "priority": priority,
            "due_date": due_date.strip(),
            "status": "Pending",
            "created_at": self._get_timestamp(),
            "completed_at": None,
        }

        tasks.append(new_task)
        self._save_tasks(tasks)

        return True, f"Task '{title}' added successfully!", task_id

    def view_all_tasks(self):
        """
        Get all tasks

        Returns:
            list: List of all tasks
        """
        return self._load_tasks()

    def view_task_by_id(self, task_id):
        """
        Get a specific task by ID

        Args:
            task_id (int): Task ID to retrieve

        Returns:
            dict or None: Task dictionary if found, None otherwise
        """
        tasks = self._load_tasks()
        for task in tasks:
            if task["id"] == task_id:
                return task
        return None

    def view_tasks_by_status(self, status):
        """
        Get tasks filtered by status

        Args:
            status (str): Status to filter by ('Pending' or 'Completed')

        Returns:
            list: List of tasks with the specified status
        """
        tasks = self._load_tasks()
        return [task for task in tasks if task["status"] == status]

    def view_tasks_by_priority(self, priority):
        """
        Get tasks filtered by priority

        Args:
            priority (str): Priority to filter by (Low/Medium/High)

        Returns:
            list: List of tasks with the specified priority
        """
        tasks = self._load_tasks()
        return [task for task in tasks if task["priority"] == priority]

    def update_task(self, task_id, **kwargs):
        """
        Update an existing task

        Args:
            task_id (int): ID of task to update
            **kwargs: Fields to update (title, description, priority, due_date)

        Returns:
            tuple: (success: bool, message: str)
        """
        tasks = self._load_tasks()
        task_found = False

        for task in tasks:
            if task["id"] == task_id:
                task_found = True

                # Update allowed fields
                if "title" in kwargs and kwargs["title"].strip():
                    task["title"] = kwargs["title"].strip()

                if "description" in kwargs:
                    task["description"] = kwargs["description"].strip()

                if "priority" in kwargs and kwargs["priority"] in [
                    "Low",
                    "Medium",
                    "High",
                ]:
                    task["priority"] = kwargs["priority"]

                if "due_date" in kwargs:
                    task["due_date"] = kwargs["due_date"].strip()

                break

        if not task_found:
            return False, f"Task with ID {task_id} not found"

        self._save_tasks(tasks)
        return True, f"Task {task_id} updated successfully!"

    def mark_complete(self, task_id):
        """
        Mark a task as completed

        Args:
            task_id (int): ID of task to mark as complete

        Returns:
            tuple: (success: bool, message: str)
        """
        tasks = self._load_tasks()
        task_found = False

        for task in tasks:
            if task["id"] == task_id:
                task_found = True

                if task["status"] == "Completed":
                    return False, f"Task {task_id} is already completed"

                task["status"] = "Completed"
                task["completed_at"] = self._get_timestamp()
                break

        if not task_found:
            return False, f"Task with ID {task_id} not found"

        self._save_tasks(tasks)
        return True, f"Task {task_id} marked as completed!"

    def mark_pending(self, task_id):
        """
        Mark a task as pending

        Args:
            task_id (int): ID of task to mark as pending

        Returns:
            tuple: (success: bool, message: str)
        """
        tasks = self._load_tasks()
        task_found = False

        for task in tasks:
            if task["id"] == task_id:
                task_found = True

                if task["status"] == "Pending":
                    return False, f"Task {task_id} is already pending"

                task["status"] = "Pending"
                task["completed_at"] = None
                break

        if not task_found:
            return False, f"Task with ID {task_id} not found"

        self._save_tasks(tasks)
        return True, f"Task {task_id} marked as pending!"

    def delete_task(self, task_id):
        """
        Delete a task

        Args:
            task_id (int): ID of task to delete

        Returns:
            tuple: (success: bool, message: str)
        """
        tasks = self._load_tasks()
        initial_count = len(tasks)

        tasks = [task for task in tasks if task["id"] != task_id]

        if len(tasks) == initial_count:
            return False, f"Task with ID {task_id} not found"

        self._save_tasks(tasks)
        return True, f"Task {task_id} deleted successfully!"

    def get_task_count(self):
        """
        Get task count statistics

        Returns:
            dict: Dictionary with task counts
        """
        tasks = self._load_tasks()
        return {
            "total": len(tasks),
            "pending": len([t for t in tasks if t["status"] == "Pending"]),
            "completed": len([t for t in tasks if t["status"] == "Completed"]),
        }

    def search_tasks(self, keyword):
        """
        Search tasks by keyword in title or description

        Args:
            keyword (str): Keyword to search for

        Returns:
            list: List of matching tasks
        """
        tasks = self._load_tasks()
        keyword_lower = keyword.lower()

        matching_tasks = []
        for task in tasks:
            if (
                keyword_lower in task["title"].lower()
                or keyword_lower in task["description"].lower()
            ):
                matching_tasks.append(task)

        return matching_tasks
