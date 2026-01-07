# # Example demo DB for NL2SQL
# # Run this script to create a sample SQLite DB for testing
# import sqlite3
# import os

# db_path = os.path.join(os.path.dirname(__file__), "data", "demo.db")
# os.makedirs(os.path.dirname(db_path), exist_ok=True)
# conn = sqlite3.connect(db_path)
# c = conn.cursor()
# c.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)")
# c.execute("CREATE TABLE IF NOT EXISTS departments (id INTEGER PRIMARY KEY, name TEXT)")
# c.execute("DELETE FROM employees")
# c.execute("DELETE FROM departments")
# c.executemany("INSERT INTO employees (id, name, department, salary) VALUES (?, ?, ?, ?)", [
#     (1, "Alice", "HR", 70000),
#     (2, "Bob", "Engineering", 90000),
#     (3, "Charlie", "Engineering", 85000),
#     (4, "Diana", "Marketing", 65000),
# ])
# c.executemany("INSERT INTO departments (id, name) VALUES (?, ?)", [
#     (1, "HR"),
#     (2, "Engineering"),
#     (3, "Marketing"),
# ])
# conn.commit()
# print(f"Demo DB created at {db_path}")
