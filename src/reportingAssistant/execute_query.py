from NL2SQL.schema import GraphState
from NL2SQL.error_handling import explain_query_error
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text, inspect
from urllib.parse import quote_plus
from langchain_core.runnables import RunnableConfig

import re
import os

def execute_query(state: GraphState, config: RunnableConfig) -> GraphState:
    """Node 2: Execute the generated SQL query."""

    all_turns = state.get("generated_query", [])
    
    query = ""
    if all_turns and all_turns[-1]:
        query = all_turns[-1][-1]
    
    if not query:
        state["error_message"] = (state.get("error_message") or []) + ["No SQL query generated"]
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
