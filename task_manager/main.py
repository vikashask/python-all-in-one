"""
Task Manager with User Authentication
Main application entry point
"""

import os
import sys
from user import UserAuth
from task import TaskManager


class TaskManagerApp:
    """Main application class for Task Manager"""

    def __init__(self):
        """Initialize the Task Manager application"""
        self.auth = UserAuth()
        self.current_user = None
        self.task_manager = None

    def clear_screen(self):
        """Clear the console screen"""
        os.system("clear" if os.name == "posix" else "cls")

    def print_header(self, title):
        """
        Print a formatted header

        Args:
            title (str): Header title
        """
        print("\n" + "=" * 60)
        print(f"{title.center(60)}")
        print("=" * 60)

    def print_separator(self):
        """Print a separator line"""
        print("-" * 60)

    def get_input(self, prompt, required=True):
        """
        Get user input with validation

        Args:
            prompt (str): Input prompt
            required (bool): Whether input is required

        Returns:
            str: User input
        """
        while True:
            value = input(prompt).strip()
            if not required or value:
                return value
            print("This field is required. Please try again.")

    def press_enter_to_continue(self):
        """Wait for user to press enter"""
        input("\nPress Enter to continue...")

    def display_main_menu(self):
        """Display main menu for unauthenticated users"""
        self.clear_screen()
        self.print_header("TASK MANAGER - WELCOME")
        print("\n1. Login")
        print("2. Register")
        print("3. Exit")
        self.print_separator()

    def display_task_menu(self):
        """Display menu for authenticated users"""
        self.clear_screen()
        self.print_header(f"TASK MANAGER - Welcome, {self.current_user}!")

        # Display task statistics
        stats = self.task_manager.get_task_count()
        print(
            f"\nTotal Tasks: {stats['total']} | "
            f"Pending: {stats['pending']} | "
            f"Completed: {stats['completed']}"
        )
        self.print_separator()

        print("\n--- Task Operations ---")
        print("1. Add New Task")
        print("2. View All Tasks")
        print("3. View Pending Tasks")
        print("4. View Completed Tasks")
        print("5. Search Tasks")
        print("6. Update Task")
        print("7. Mark Task as Complete")
        print("8. Mark Task as Pending")
        print("9. Delete Task")
        print("\n--- Other Options ---")
        print("10. Logout")
        self.print_separator()

    def register_user(self):
        """Handle user registration"""
        self.clear_screen()
        self.print_header("USER REGISTRATION")

        print("\nUsername requirements:")
        print("- At least 3 characters long")
        print("- Only letters, numbers, and underscores allowed")
        print("\nPassword requirements:")
        print("- At least 6 characters long")
        self.print_separator()

        username = self.get_input("\nEnter username: ")
        password = self.get_input("Enter password: ")
        confirm_password = self.get_input("Confirm password: ")

        if password != confirm_password:
            print("\n❌ Passwords do not match!")
            self.press_enter_to_continue()
            return

        success, message = self.auth.register(username, password)

        if success:
            print(f"\n✅ {message}")
            print("You can now login with your credentials.")
        else:
            print(f"\n❌ {message}")

        self.press_enter_to_continue()

    def login_user(self):
        """Handle user login"""
        self.clear_screen()
        self.print_header("USER LOGIN")

        username = self.get_input("\nEnter username: ")
        password = self.get_input("Enter password: ")

        success, message = self.auth.login(username, password)

        if success:
            self.current_user = username
            self.task_manager = TaskManager(username)
            print(f"\n✅ {message}")
            print(f"Welcome back, {username}!")
            self.press_enter_to_continue()
            return True
        else:
            print(f"\n❌ {message}")
            self.press_enter_to_continue()
            return False

    def logout_user(self):
        """Handle user logout"""
        self.current_user = None
        self.task_manager = None
        print("\n✅ Logged out successfully!")
        self.press_enter_to_continue()

    def add_task(self):
        """Add a new task"""
        self.clear_screen()
        self.print_header("ADD NEW TASK")

        title = self.get_input("\nTask Title: ")
        description = self.get_input("Task Description (optional): ", required=False)

        print("\nPriority levels: Low, Medium, High")
        priority = self.get_input("Priority (default: Medium): ", required=False)
        if not priority:
            priority = "Medium"

        due_date = self.get_input(
            "Due Date (e.g., 2025-12-31) (optional): ", required=False
        )

        success, message, task_id = self.task_manager.add_task(
            title, description, priority, due_date
        )

        if success:
            print(f"\n✅ {message}")
            print(f"Task ID: {task_id}")
        else:
            print(f"\n❌ {message}")

        self.press_enter_to_continue()

    def display_task(self, task, show_index=False, index=None):
        """
        Display a single task

        Args:
            task (dict): Task dictionary
            show_index (bool): Whether to show task index
            index (int): Task index
        """
        status_symbol = "✅" if task["status"] == "Completed" else "⏳"

        if show_index:
            print(f"\n{index}. {status_symbol} Task ID: {task['id']}")
        else:
            print(f"\n{status_symbol} Task ID: {task['id']}")

        print(f"   Title: {task['title']}")

        if task["description"]:
            print(f"   Description: {task['description']}")

        print(f"   Priority: {task['priority']}")
        print(f"   Status: {task['status']}")

        if task["due_date"]:
            print(f"   Due Date: {task['due_date']}")

        print(f"   Created: {task['created_at']}")

        if task["completed_at"]:
            print(f"   Completed: {task['completed_at']}")

    def view_all_tasks(self):
        """View all tasks"""
        self.clear_screen()
        self.print_header("ALL TASKS")

        tasks = self.task_manager.view_all_tasks()

        if not tasks:
            print("\nNo tasks found. Start by adding a new task!")
        else:
            print(f"\nTotal: {len(tasks)} task(s)")
            self.print_separator()
            for i, task in enumerate(tasks, 1):
                self.display_task(task, show_index=True, index=i)

        self.press_enter_to_continue()

    def view_pending_tasks(self):
        """View pending tasks"""
        self.clear_screen()
        self.print_header("PENDING TASKS")

        tasks = self.task_manager.view_tasks_by_status("Pending")

        if not tasks:
            print("\n🎉 No pending tasks! You're all caught up!")
        else:
            print(f"\nTotal: {len(tasks)} pending task(s)")
            self.print_separator()
            for i, task in enumerate(tasks, 1):
                self.display_task(task, show_index=True, index=i)

        self.press_enter_to_continue()

    def view_completed_tasks(self):
        """View completed tasks"""
        self.clear_screen()
        self.print_header("COMPLETED TASKS")

        tasks = self.task_manager.view_tasks_by_status("Completed")

        if not tasks:
            print("\nNo completed tasks yet. Keep working on your tasks!")
        else:
            print(f"\nTotal: {len(tasks)} completed task(s)")
            self.print_separator()
            for i, task in enumerate(tasks, 1):
                self.display_task(task, show_index=True, index=i)

        self.press_enter_to_continue()

    def search_tasks(self):
        """Search for tasks"""
        self.clear_screen()
        self.print_header("SEARCH TASKS")

        keyword = self.get_input("\nEnter search keyword: ")
        tasks = self.task_manager.search_tasks(keyword)

        if not tasks:
            print(f"\nNo tasks found matching '{keyword}'")
        else:
            print(f"\nFound {len(tasks)} task(s) matching '{keyword}':")
            self.print_separator()
            for i, task in enumerate(tasks, 1):
                self.display_task(task, show_index=True, index=i)

        self.press_enter_to_continue()

    def update_task(self):
        """Update an existing task"""
        self.clear_screen()
        self.print_header("UPDATE TASK")

        try:
            task_id = int(self.get_input("\nEnter Task ID to update: "))
        except ValueError:
            print("\n❌ Invalid Task ID. Please enter a number.")
            self.press_enter_to_continue()
            return

        task = self.task_manager.view_task_by_id(task_id)

        if not task:
            print(f"\n❌ Task with ID {task_id} not found")
            self.press_enter_to_continue()
            return

        print("\nCurrent task details:")
        self.display_task(task)

        print("\n\nEnter new values (press Enter to keep current value):")
        self.print_separator()

        new_title = self.get_input(f"New Title [{task['title']}]: ", required=False)
        new_description = self.get_input(
            f"New Description [{task['description']}]: ", required=False
        )
        new_priority = self.get_input(
            f"New Priority [{task['priority']}]: ", required=False
        )
        new_due_date = self.get_input(
            f"New Due Date [{task['due_date']}]: ", required=False
        )

        updates = {}
        if new_title:
            updates["title"] = new_title
        if new_description or new_description == "":
            updates["description"] = new_description
        if new_priority:
            updates["priority"] = new_priority
        if new_due_date or new_due_date == "":
            updates["due_date"] = new_due_date

        if not updates:
            print("\nNo changes made.")
            self.press_enter_to_continue()
            return

        success, message = self.task_manager.update_task(task_id, **updates)

        if success:
            print(f"\n✅ {message}")
        else:
            print(f"\n❌ {message}")

        self.press_enter_to_continue()

    def mark_task_complete(self):
        """Mark a task as complete"""
        self.clear_screen()
        self.print_header("MARK TASK AS COMPLETE")

        try:
            task_id = int(self.get_input("\nEnter Task ID to mark as complete: "))
        except ValueError:
            print("\n❌ Invalid Task ID. Please enter a number.")
            self.press_enter_to_continue()
            return

        success, message = self.task_manager.mark_complete(task_id)

        if success:
            print(f"\n✅ {message}")
        else:
            print(f"\n❌ {message}")

        self.press_enter_to_continue()

    def mark_task_pending(self):
        """Mark a task as pending"""
        self.clear_screen()
        self.print_header("MARK TASK AS PENDING")

        try:
            task_id = int(self.get_input("\nEnter Task ID to mark as pending: "))
        except ValueError:
            print("\n❌ Invalid Task ID. Please enter a number.")
            self.press_enter_to_continue()
            return

        success, message = self.task_manager.mark_pending(task_id)

        if success:
            print(f"\n✅ {message}")
        else:
            print(f"\n❌ {message}")

        self.press_enter_to_continue()

    def delete_task(self):
        """Delete a task"""
        self.clear_screen()
        self.print_header("DELETE TASK")

        try:
            task_id = int(self.get_input("\nEnter Task ID to delete: "))
        except ValueError:
            print("\n❌ Invalid Task ID. Please enter a number.")
            self.press_enter_to_continue()
            return

        task = self.task_manager.view_task_by_id(task_id)

        if not task:
            print(f"\n❌ Task with ID {task_id} not found")
            self.press_enter_to_continue()
            return

        print("\nTask to delete:")
        self.display_task(task)

        confirm = self.get_input(
            "\nAre you sure you want to delete this task? (yes/no): "
        )

        if confirm.lower() in ["yes", "y"]:
            success, message = self.task_manager.delete_task(task_id)

            if success:
                print(f"\n✅ {message}")
            else:
                print(f"\n❌ {message}")
        else:
            print("\n❌ Task deletion cancelled.")

        self.press_enter_to_continue()

    def run_main_menu(self):
        """Run the main menu loop for unauthenticated users"""
        while True:
            self.display_main_menu()
            choice = self.get_input("Enter your choice (1-3): ")

            if choice == "1":
                if self.login_user():
                    self.run_task_menu()
            elif choice == "2":
                self.register_user()
            elif choice == "3":
                self.clear_screen()
                print("\n👋 Thank you for using Task Manager. Goodbye!")
                sys.exit(0)
            else:
                print("\n❌ Invalid choice. Please try again.")
                self.press_enter_to_continue()

    def run_task_menu(self):
        """Run the task menu loop for authenticated users"""
        while self.current_user:
            self.display_task_menu()
            choice = self.get_input("Enter your choice (1-10): ")

            if choice == "1":
                self.add_task()
            elif choice == "2":
                self.view_all_tasks()
            elif choice == "3":
                self.view_pending_tasks()
            elif choice == "4":
                self.view_completed_tasks()
            elif choice == "5":
                self.search_tasks()
            elif choice == "6":
                self.update_task()
            elif choice == "7":
                self.mark_task_complete()
            elif choice == "8":
                self.mark_task_pending()
            elif choice == "9":
                self.delete_task()
            elif choice == "10":
                self.logout_user()
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
                self.press_enter_to_continue()

    def run(self):
        """Start the application"""
        self.run_main_menu()


def main():
    """Main entry point"""
    try:
        app = TaskManagerApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
