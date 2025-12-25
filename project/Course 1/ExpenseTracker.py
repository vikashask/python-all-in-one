import csv
import os
from datetime import datetime
from typing import List, Dict, Optional


class ExpenseTracker:

    def __init__(self, csv_filename: str = "expenses.csv"):
        self.expenses: List[Dict[str, str]] = []
        self.monthly_budget: float = 0.0
        self.csv_filename = csv_filename
        self.load_expenses()

    def add_expense(self) -> None:
        print("\n--- Add New Expense ---")
        # Get and validate date
        while True:
            date_input = input("Enter date (YYYY-MM-DD): ").strip()
            if self._validate_date(date_input):
                break
            print("Invalid date format. Please use YYYY-MM-DD format.")

        # Get category
        while True:
            category = input(
                "Enter category (e.g., Food, Travel, Entertainment): "
            ).strip()
            if category:
                break
            print("Category cannot be empty.")

        # Get and validate amount
        while True:
            try:
                amount = float(input("Enter amount: ₹").strip())
                if amount > 0:
                    break
                print("Amount must be positive.")
            except ValueError:
                print("Please enter a valid number.")

        # Get description
        while True:
            description = input("Enter description: ").strip()
            if description:
                break
            print("Description cannot be empty.")

        # Create expense dictionary and add to list
        expense = {
            "date": date_input,
            "category": category,
            "amount": amount,
            "description": description,
        }

        self.expenses.append(expense)
        print(f"✓ Expense of ₹{amount:.2f} for {category} added successfully!")

    def view_expenses(self) -> None:
        print("\n--- Your Expenses ---")

        if not self.expenses:
            print("No expenses recorded yet.")
            return

        # Header
        print(f"{'Date':<12} {'Category':<15} {'Amount':<10} {'Description':<30}")
        print("-" * 70)

        valid_expenses = 0
        total_amount = 0

        for expense in self.expenses:
            # Validate expense data
            if self._validate_expense(expense):
                print(
                    f"{expense['date']:<12} {expense['category']:<15} "
                    f"₹{expense['amount']:<9.2f} {expense['description']:<30}"
                )
                valid_expenses += 1
                total_amount += expense["amount"]
            else:
                print(f"⚠️  Incomplete expense entry skipped")

        print("-" * 70)
        print(f"Total valid expenses: {valid_expenses}")
        print(f"Total amount: ₹{total_amount:.2f}")

    def set_budget(self) -> None:
        print("\n--- Set Monthly Budget ---")

        while True:
            try:
                budget = float(input("Enter your monthly budget: ₹").strip())
                if budget > 0:
                    self.monthly_budget = budget
                    print(f"✓ Monthly budget set to ₹{budget:.2f}")
                    break
                print("Budget must be positive.")
            except ValueError:
                print("Please enter a valid number.")

    def track_budget(self) -> None:
        print("\n--- Budget Tracking ---")

        if self.monthly_budget <= 0:
            print("No budget set. Please set a monthly budget first.")
            self.set_budget()
            return

        # Calculate total expenses for valid entries
        total_expenses = sum(
            expense["amount"]
            for expense in self.expenses
            if self._validate_expense(expense)
        )

        print(f"Monthly Budget: ₹{self.monthly_budget:.2f}")
        print(f"Total Expenses: ₹{total_expenses:.2f}")

        if total_expenses > self.monthly_budget:
            overspent = total_expenses - self.monthly_budget
            print(f"⚠️  WARNING: You have exceeded your budget by ₹{overspent:.2f}!")
        else:
            remaining = self.monthly_budget - total_expenses
            print(f"✓ You have ₹{remaining:.2f} left for the month")

        # Calculate percentage spent
        percentage = (total_expenses / self.monthly_budget) * 100
        print(f"Budget used: {percentage:.1f}%")

    def save_expenses(self) -> None:
        try:
            with open(self.csv_filename, "w", newline="", encoding="utf-8") as file:
                if self.expenses:
                    # Write header and data
                    fieldnames = ["date", "category", "amount", "description"]
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()

                    valid_count = 0
                    for expense in self.expenses:
                        if self._validate_expense(expense):
                            writer.writerow(expense)
                            valid_count += 1

                    print(f"✓ {valid_count} expenses saved to {self.csv_filename}")
                else:
                    # Create empty file with headers
                    writer = csv.writer(file)
                    writer.writerow(["date", "category", "amount", "description"])
                    print(f"✓ Empty expense file created: {self.csv_filename}")

        except Exception as e:
            print(f"❌ Error saving expenses: {e}")

    def load_expenses(self) -> None:
        if not os.path.exists(self.csv_filename):
            print(f"No existing expense file found. Starting fresh.")
            return

        try:
            with open(self.csv_filename, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                loaded_expenses = []

                for row in reader:
                    # Convert amount back to float
                    try:
                        row["amount"] = float(row["amount"])
                        loaded_expenses.append(row)
                    except (ValueError, KeyError) as e:
                        print(f"⚠️  Skipping invalid row: {e}")

                self.expenses = loaded_expenses
                print(
                    f"✓ Loaded {len(self.expenses)} expenses from {self.csv_filename}"
                )

        except Exception as e:
            print(f"❌ Error loading expenses: {e}")

    def display_menu(self) -> None:
        """
        Display the main menu options.
        """
        print("\n" + "=" * 50)
        print("      PERSONAL EXPENSE TRACKER")
        print("=" * 50)
        print("1. Add expense")
        print("2. View expenses")
        print("3. Track budget")
        print("4. Save expenses")
        print("5. Exit")
        print("-" * 50)

    def run(self) -> None:
        print("Welcome to Personal Expense Tracker!")

        while True:
            self.display_menu()

            try:
                choice = input("Enter your choice (1-5): ").strip()

                if choice == "1":
                    self.add_expense()
                elif choice == "2":
                    self.view_expenses()
                elif choice == "3":
                    self.track_budget()
                elif choice == "4":
                    self.save_expenses()
                elif choice == "5":
                    print("\nSaving expenses before exit...")
                    self.save_expenses()
                    print("Thank you for using Personal Expense Tracker!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")

            except KeyboardInterrupt:
                print("\n\nProgram interrupted. Saving expenses...")
                self.save_expenses()
                print("Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

    def _validate_date(self, date_str: str) -> bool:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _validate_expense(self, expense: Dict) -> bool:

        required_fields = ["date", "category", "amount", "description"]

        # Check if all required fields exist and are not empty
        for field in required_fields:
            if field not in expense or not expense[field]:
                return False

        # Validate date format
        if not self._validate_date(expense["date"]):
            return False

        # Validate amount is a number
        try:
            float(expense["amount"])
        except (ValueError, TypeError):
            return False

        return True


# Example usage and main execution
if __name__ == "__main__":
    # Create and run the expense tracker
    tracker = ExpenseTracker()
    tracker.run()
