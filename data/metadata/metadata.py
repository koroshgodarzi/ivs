import os
import json
import warnings
from sqlalchemy import create_engine, inspect, exc, text

# Suppress the server version warning
warnings.filterwarnings("ignore", category=exc.SAWarning)

def save_view_schemas_to_json():
    # 1. Connection configuration
    sql_server = os.getenv("SQL_SERVER", "localhost")
    sql_database = os.getenv("SQL_DATABASE", "IPMPBI")
    sql_user = os.getenv("SQL_USER", "sa")
    sql_password = os.getenv("SQL_PASSWORD", "YourStrongPassword123!")
    sql_port = int(os.getenv("SQL_PORT", "1433"))

    connection_uri = f"mssql+pytds://{sql_user}:{sql_password}@{sql_server}:{sql_port}/{sql_database}"
    engine = create_engine(connection_uri)

    output_dir = "."

    try:
        inspector = inspect(engine)
        schema_name = 'Report'
        views = inspector.get_view_names(schema=schema_name)
        
        if not views:
            print(f"No views found in schema '{schema_name}'.")
            return

        print(f"Found {len(views)} views. Extracting schemas and data samples...")

        # Open connection once to perform data queries
        with engine.connect() as conn:
            for view_name in views:
                columns_raw = inspector.get_columns(view_name, schema=schema_name)
                view_data = []

                for col in columns_raw:
                    col_name = col['name']
                    
                    # Logic to fetch unique values
                    # We fetch TOP 6 to see if it exceeds our threshold of 5
                    try:
                        # Use quoted identifiers for schema, view, and column names to handle spaces/Persian chars
                        query = text(f'SELECT DISTINCT TOP 100 "{col_name}" FROM "{schema_name}"."{view_name}"')
                        result = conn.execute(query).fetchall()
                        
                        # Extract values and convert to string if not JSON serializable (Date, Decimal, etc.)
                        raw_values = [row[0] for row in result if row[0] is not None]
                        
                        # Process logic: 5 or less = unique_values, more than 5 = example_values (2)
                        val_info = {}
                        if len(raw_values) <= 10:
                            val_info["unique_values"] = [str(v) if not isinstance(v, (int, float, bool)) else v for v in raw_values]
                        else:
                            # Take only 2 as examples
                            val_info["example_values"] = [str(v) if not isinstance(v, (int, float, bool)) else v for v in raw_values[:2]]
                    
                    except Exception as col_err:
                        # Some columns (like image/text/blob) might not support DISTINCT
                        val_info = {"error": "Could not fetch values"}
                        print(f"   ! Could not fetch values for {view_name}.{col_name}: {col_err}")

                    column_entry = {
                        "name": col_name,
                        "type": str(col['type']),
                        "nullable": col['nullable'],
                        "default": str(col['default']) if col.get('default') else None,
                    }
                    # Merge the unique/example value info into the column dictionary
                    column_entry.update(val_info)
                    view_data.append(column_entry)

                # Save to JSON
                file_path = os.path.join(output_dir, f"{view_name}_column_meta.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(view_data, f, indent=4, ensure_ascii=False)
                
                print(f" - Saved: {file_path}")

        print(f"\nSuccess! All files are in the '{output_dir}' folder.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    save_view_schemas_to_json()