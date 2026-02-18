from graph.schema import GraphState
from graph.error_handling import explain_query_error
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text, inspect
from urllib.parse import quote_plus
from graph.utils import fix_sql_wildcards
from langchain_core.runnables import RunnableConfig

import re
import os

def execute_query(state: GraphState, config: RunnableConfig) -> GraphState:
    """Node 2: Execute the generated SQL query."""

    queries = [q for q in state.get("generated_query", [])]
    query = queries[-1] if queries else ""
    query = fix_sql_wildcards(query)
    print(f"Here on executing: {query}")
    
    if not query:
        state["error_message"] = "No SQL query generated"
        state["query_results"] = None
        return state
    
    results, error = run_sql_server_query(query)
    
    if error:
        errors = state.get("error_message") or []
        errors.append(error)
        state["error_message"] = errors
    else:
        state["query_results"] = results
        print(f"Query result: {results}")

    return state


def run_sql_server_query(
    query: str,
    server: str = os.getenv("SQL_SERVER", "localhost"),
    database: str = os.getenv("SQL_DATABASE", "IPMPBI"),
    user: str = os.getenv("SQL_USER", "sa"),
    password: str = os.getenv("SQL_PASSWORD", "YourStrongPassword123!"),
    port: int = int(os.getenv("SQL_PORT", "1433"))
) -> tuple:
    """
    Execute a SQL query against a SQL Server database using SQLAlchemy and pytds.
    """

    if not is_select_only(query):
        return None, "Only SELECT queries are allowed."

    try:
        # 1. Prepare credentials for the URL
        # quote_plus handles special characters in passwords
        safe_password = quote_plus(password)
        
        connection_url = (
            f"mssql+pytds://{user}:{safe_password}@{server}:{port}/{database}"
        )
        engine = create_engine(connection_url, future=True)

        with engine.connect() as conn:
            executable_query = text(query)            
            result = conn.execute(executable_query)
            rows = result.mappings().all()
            results = []
            for row in rows:
                row_dict = {}
                for key, value in row.items():
                    # Handle common non-serializable types
                    if value is None:
                        row_dict[key] = None
                    elif isinstance(value, (str, int, float, bool)):
                        row_dict[key] = value
                    else:
                        # Convert Decimals, DateTimes, etc. to strings
                        row_dict[key] = str(value)
                results.append(row_dict)
            
            return results, None

    except SQLAlchemyError as e:
        return None, f"Database error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
    finally:
        if 'engine' in locals():
            engine.dispose()


def is_select_only(query: str) -> bool:
    """
    Validate that the query only contains SELECT statements.
    Blocks DROP, DELETE, INSERT, UPDATE, and other dangerous operations.
    """
    query_normalized = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
    query_normalized = re.sub(r'/\*.*?\*/', '', query_normalized, flags=re.DOTALL)
    query_normalized = query_normalized.strip().upper()
    
    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 
        'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT',
        'REVOKE', 'MERGE', 'REPLACE'
    ]
    
    for keyword in dangerous_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, query_normalized):
            return False
    
    if not query_normalized.startswith('SELECT'):
        return False
    
    return True


if __name__ == "__main__":
    print("=== Testing SQL Execution Node ===\n")

    # 1. Setup Mock Environment Variables (Change these to your real DB to test connection)
    os.environ["SQL_SERVER"] = "localhost"
    os.environ["SQL_DATABASE"] = "IPMPBI"
    os.environ["SQL_USER"] = "sa"
    os.environ["SQL_PASSWORD"] = "TestPassword123!"

    # 2. Test Security Logic (is_select_only)
    print("--- Testing Security Filter ---")
    queries = [
        ("SELECT * FROM Employees", True),
        ("UPDATE Employees SET Salary = 100000", False),
        ("SELECT * FROM Users; DROP TABLE Users", False),
        ("  -- comment \n SELECT name FROM products", True),
        ("EXEC sp_whoisactive", False)
    ]

    for q, expected in queries:
        result = is_select_only(q)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Query: {q[:40]}... -> Allowed: {result}")

    # 3. Test execute_query with a Malicious Query (State Flow)
    print("\n--- Testing State Flow: Malicious Query ---")
    state_malicious: GraphState = {
        "messages": [{"role": "user", "content": "Delete all my data"}],
        "generated_query": "DELETE FROM Users",
        "query_results": None,
        "error_message": None
    }
    
    result_state = execute_query(state_malicious)
    print(f"Error Message: {result_state.get('error_message')}")

    # 4. Test execute_query with a Valid Query (Database Interaction)
    # Note: This will attempt to connect to a real DB. 
    # If no DB is running, it will correctly catch the SQLAlchemyError.
    print("\n--- Testing State Flow: Database Connection ---")
    state_valid: GraphState = {
        "messages": [{"role": "user", "content": "Show me all projects"}],
        "generated_query": "SELECT TOP 5 ProjectName FROM Projects",
        "query_results": None,
        "error_message": None
    }

    final_state = execute_query(state_valid)

    if final_state["query_results"]:
        print("Success! Data retrieved:")
        print(final_state["query_results"])
    else:
        print("Note: Database connection failed as expected (no live DB).")
        print(f"Captured Error: {final_state['error_message']}")

    print("\n=== Testing Complete ===")