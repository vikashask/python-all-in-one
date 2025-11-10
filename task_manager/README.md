# Task Manager with User Authentication

A comprehensive command-line Task Manager application built with core Python that allows multiple users to manage their tasks with secure authentication.

## Features

### User Authentication

- **User Registration**: Create new accounts with username and password
- **Secure Login**: Password hashing using SHA-256
- **User Isolation**: Each user has their own separate task list

### Task Management

- **Add Tasks**: Create tasks with title, description, priority, and due date
- **View Tasks**: Display all tasks, pending tasks, or completed tasks
- **Update Tasks**: Modify existing task details
- **Mark Complete/Pending**: Change task status
- **Delete Tasks**: Remove unwanted tasks
- **Search Tasks**: Find tasks by keyword in title or description
- **Task Statistics**: View counts of total, pending, and completed tasks

### Data Persistence

- User credentials stored securely in `data/users.json`
- Individual task files for each user in `data/tasks/`
- All data persists between sessions

## Project Structure

```
Task Manager/
├── main.py                 # Main application entry point
├── user.py                 # User authentication module
├── task.py                 # Task management module
├── data/                   # Data storage directory
│   ├── users.json         # User credentials (auto-created)
│   └── tasks/             # User task files (auto-created)
│       └── <username>_tasks.json
└── README.md              # This file
```

## Requirements

- Python 3.6 or higher
- No external dependencies (uses only core Python libraries)

## Installation

1. Clone or download this project to your local machine
2. Navigate to the project directory:
   ```bash
   cd "Task Manager"
   ```

## Usage

### Running the Application

Start the application by running:

```bash
python3 main.py
```

### First Time Setup

1. **Register an Account**

   - Choose option `2` from the main menu
   - Enter a username (min 3 characters, alphanumeric + underscores)
   - Enter a password (min 6 characters)
   - Confirm your password

2. **Login**
   - Choose option `1` from the main menu
   - Enter your username and password

### Managing Tasks

Once logged in, you can:

#### 1. Add New Task

- Choose option `1`
- Enter task details:
  - Title (required)
  - Description (optional)
  - Priority: Low, Medium, or High (default: Medium)
  - Due Date (optional, e.g., 2025-12-31)

#### 2. View All Tasks

- Choose option `2`
- Displays all your tasks with full details

#### 3. View Pending Tasks

- Choose option `3`
- Shows only tasks that are not yet completed

#### 4. View Completed Tasks

- Choose option `4`
- Shows only completed tasks

#### 5. Search Tasks

- Choose option `5`
- Enter a keyword to search in task titles and descriptions

#### 6. Update Task

- Choose option `6`
- Enter the Task ID
- Update any field (press Enter to keep current value)

#### 7. Mark Task as Complete

- Choose option `7`
- Enter the Task ID
- Task status changes to "Completed"

#### 8. Mark Task as Pending

- Choose option `8`
- Enter the Task ID
- Task status changes back to "Pending"

#### 9. Delete Task

- Choose option `9`
- Enter the Task ID
- Confirm deletion

#### 10. Logout

- Choose option `10`
- Returns to main menu

## Example Workflow

```
1. Register a new account: username "alice", password "alice123"
2. Login with credentials
3. Add a task: "Complete Python project"
4. Add another task: "Review code documentation"
5. View all tasks to see your task list
6. Mark task #1 as complete
7. View pending tasks
8. Search for "Python" keyword
9. Update task #2 with new details
10. Logout
```

## Data Storage

### User Credentials

Stored in `data/users.json`:

```json
{
  "alice": {
    "password": "hashed_password_here",
    "created_at": "2025-11-09T10:30:00"
  }
}
```

### User Tasks

Each user has a separate file: `data/tasks/alice_tasks.json`:

```json
[
  {
    "id": 1,
    "title": "Complete Python project",
    "description": "Finish the task manager application",
    "priority": "High",
    "due_date": "2025-11-15",
    "status": "Completed",
    "created_at": "2025-11-09 10:35:00",
    "completed_at": "2025-11-09 11:00:00"
  }
]
```

## Security Features

- **Password Hashing**: Passwords are hashed using SHA-256 before storage
- **User Isolation**: Each user can only access their own tasks
- **Input Validation**: All user inputs are validated
- **Secure Storage**: Credentials and tasks stored in separate files

## Error Handling

The application handles various errors gracefully:

- Invalid credentials during login
- Duplicate usernames during registration
- Non-existent task IDs
- Invalid input formats
- File I/O errors

## Technical Details

### Modules

#### `user.py` - UserAuth Class

- `register(username, password)`: Create new user account
- `login(username, password)`: Authenticate user
- `user_exists(username)`: Check if user exists
- `get_all_usernames()`: List all registered users

#### `task.py` - TaskManager Class

- `add_task(title, description, priority, due_date)`: Create new task
- `view_all_tasks()`: Get all tasks
- `view_tasks_by_status(status)`: Filter by status
- `view_tasks_by_priority(priority)`: Filter by priority
- `update_task(task_id, **kwargs)`: Update task fields
- `mark_complete(task_id)`: Mark as completed
- `mark_pending(task_id)`: Mark as pending
- `delete_task(task_id)`: Remove task
- `search_tasks(keyword)`: Search by keyword
- `get_task_count()`: Get statistics

#### `main.py` - TaskManagerApp Class

- Menu-driven interface
- User interaction handling
- Display formatting
- Application flow control

## Tips

1. **Task IDs**: Each task has a unique ID number - use this to update, complete, or delete tasks
2. **Priority Levels**: Use Low, Medium, or High to organize task importance
3. **Search**: Use keywords to quickly find tasks
4. **Statistics**: Check the header when logged in to see your task progress
5. **Multiple Users**: Each user's data is completely separate and secure

## Troubleshooting

**Problem**: "Permission denied" error

- **Solution**: Ensure you have write permissions in the project directory

**Problem**: Tasks not saving

- **Solution**: Check if the `data/tasks/` directory exists and is writable

**Problem**: Cannot login

- **Solution**: Verify your username and password are correct (case-sensitive)

**Problem**: Application crashes

- **Solution**: Ensure you're using Python 3.6 or higher

## Future Enhancements

Possible improvements for this project:

- Task categories/tags
- Task reminders
- Export tasks to CSV/PDF
- Recurring tasks
- Task sharing between users
- Mobile or web interface
- Database integration

## License

This project is created for educational purposes as part of the SimpleLearn Task Manager project.

## Author

Created as a Core Python project demonstrating:

- Object-Oriented Programming
- File I/O operations
- User authentication
- CRUD operations
- Menu-driven interfaces
- Data persistence

---

**Happy Task Managing! 📝✨**
