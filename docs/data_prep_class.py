import os
import json
import csv
import warnings
from collections import defaultdict
import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, exc, text
import chromadb
from chromadb.utils import embedding_functions

# Suppress SQLAlchemy server version warnings
warnings.filterwarnings("ignore", category=exc.SAWarning)

class NL2SQLDataPipeline:
    def __init__(self, db_config=None, chroma_db_path='../data/schema_db', 
                 collection_name='schema_collection', excluded_csv_path=None): # Added excluded_csv_path
        """
        Initializes the NL2SQL Pipeline with database and ChromaDB settings.
        :param excluded_csv_path: Path to a CSV file containing columns to be excluded.
        """
        self.db_config = db_config or {
            "sql_server": os.getenv("SQL_SERVER", "localhost"),
            "sql_database": os.getenv("SQL_DATABASE", "IPMPBI"),
            "sql_user": os.getenv("SQL_USER", "sa"),
            "sql_password": os.getenv("SQL_PASSWORD", "YourStrongPassword123!"),
            "sql_port": int(os.getenv("SQL_PORT", "1433"))
        }
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.engine = None
        self._excluded_columns = self._load_excluded_columns(excluded_csv_path) # Load exclusions

    def _get_engine(self):
        """Creates and returns the SQLAlchemy engine singleton."""
        if not self.engine:
            connection_uri = (
                f"mssql+pytds://{self.db_config['sql_user']}:{self.db_config['sql_password']}"
                f"@{self.db_config['sql_server']}:{self.db_config['sql_port']}/{self.db_config['sql_database']}"
            )
            self.engine = create_engine(connection_uri)
        return self.engine

    def _load_excluded_columns(self, csv_file_path: str) -> set:
        """
        Loads table and column names from a CSV file into a set for quick exclusion checks.
        The set stores (table_name.lower(), column_name.lower()) for case-insensitive matching.
        """
        excluded_set = set()
        if not csv_file_path or not os.path.exists(csv_file_path):
            if csv_file_path: # Only print warning if path was provided but not found
                print(f"Warning: Exclusion CSV '{csv_file_path}' not found or path is empty. No columns will be excluded via CSV.")
            return excluded_set
        
        try:
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            
            table_col = next((c for c in df.columns if c.upper() in ['TABLE_NAME', 'TABLE', 'VIEW_NAME', 'VIEW']), None)
            column_col = next((c for c in df.columns if c.upper() in ['COLUMN_NAME', 'COLUMN', 'COL']), None)

            if not table_col or not column_col:
                print(f"Warning: Exclusion CSV '{csv_file_path}' must contain fields indicating table and column names. Columns found: {list(df.columns)}. No columns will be excluded via CSV.")
                return excluded_set

            for _, row in df.iterrows():
                table_name = str(row[table_col]).strip() if pd.notna(row[table_col]) else ""
                column_name = str(row[column_col]).strip() if pd.notna(row[column_col]) else ""
                if table_name and column_name:
                    excluded_set.add((table_name.lower(), column_name.lower())) # Store lowercased for case-insensitive match
            print(f"Loaded {len(excluded_set)} column exclusions from '{csv_file_path}'.")
        except Exception as e:
            print(f"Error loading exclusion CSV '{csv_file_path}': {e}. No columns will be excluded via CSV.")
        return excluded_set

    def convert_view_desc_csv_to_json(self, csv_file_path: str, json_output_path: str) -> dict:
        """
        Converts a CSV with view/table descriptions into a mapped dictionary and saves it as a JSON file.
        Robustly handles CSV BOMs, casing variations, and whitespaces.
        """
        try:
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        except FileNotFoundError:
            print(f"Error: The file '{csv_file_path}' was not found.")
            return {}
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return {}

        view_col = next((c for c in df.columns if c.upper() in ['VIEW_NAME', 'VIEW NAME', 'VIEW', 'TABLE_NAME', 'TABLE']), None)
        desc_col = next((c for c in df.columns if c.upper() in ['DESCRIPTION', 'DESC']), None)

        if not view_col or not desc_col:
            print(f"Error: CSV must contain fields indicating the view name and its description. Columns found: {list(df.columns)}")
            return {}

        view_desc_map = {}
        for _, row in df.iterrows():
            if pd.isna(row[view_col]):
                continue
            
            view_name = str(row[view_col]).strip()
            description = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""
            view_desc_map[view_name] = description

        try:
            output_dir = os.path.dirname(json_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(view_desc_map, f, indent=4, ensure_ascii=False)
            print(f"Successfully converted CSV and saved JSON to: '{json_output_path}'")
        except Exception as e:
            print(f"Error saving JSON to path '{json_output_path}': {e}")

        return view_desc_map

    def convert_column_desc_csv_to_jsonl(self, csv_file_path: str, jsonl_output_path: str) -> dict:
        """
        Converts a column description CSV into a structured JSONL (JSON Lines) file.
        Groups columns under their respective table/view names and skips records with empty column names
        or those present in the exclusion list.
        """
        try:
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        except FileNotFoundError:
            print(f"Error: The file '{csv_file_path}' was not found.")
            return {}
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return {}

        table_col = next((c for c in df.columns if c.upper() in ['TABLE_NAME', 'TABLE', 'VIEW_NAME', 'VIEW']), None)
        column_col = next((c for c in df.columns if c.upper() in ['COLUMN_NAME', 'COLUMN', 'COL']), None)
        desc_col = next((c for c in df.columns if c.upper() in ['DESCRIPTION', 'DESC']), None)

        if not table_col or not column_col or not desc_col:
            print(f"Error: CSV must contain fields indicating the Table Name, Column Name, and Description. Columns found: {list(df.columns)}")
            return {}

        aggregated_data = defaultdict(dict)
        for _, row in df.iterrows():
            if pd.isna(row[table_col]) or pd.isna(row[column_col]):
                continue
            
            table_name = str(row[table_col]).strip()
            column_name = str(row[column_col]).strip()

            # NEW: Exclusion check
            if (table_name.lower(), column_name.lower()) in self._excluded_columns:
                print(f"   - Skipping excluded column from CSV conversion: {table_name}.{column_name}")
                continue

            description = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""

            if not table_name or not column_name:
                continue

            aggregated_data[table_name][column_name] = description

        try:
            output_dir = os.path.dirname(jsonl_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                for table_name, columns in aggregated_data.items():
                    line_data = {
                        "view_name": table_name,
                        "columns": columns
                    }
                    f.write(json.dumps(line_data, ensure_ascii=False) + '\n')
            print(f"Successfully converted CSV and saved JSONL to: '{jsonl_output_path}'")
        except Exception as e:
            print(f"Error saving JSONL to path '{jsonl_output_path}': {e}")

        return dict(aggregated_data)

    def extract_database_metadata(self, schema_name='Report', output_dir='./metadata_output'):
        """
        Connects to the SQL Server, extracts schema data, and identifies 
        categorical values (<=10) or examples (>10). Saves metadata as JSON.
        Excludes columns specified in the exclusion list.
        """
        engine = self._get_engine()
        metadata_registry = {}

        try:
            inspector = inspect(engine)
            views = inspector.get_view_names(schema=schema_name)
            
            if not views:
                print(f"No views found in schema '{schema_name}'. Trying tables...")
                views = inspector.get_table_names(schema=schema_name)
                if not views:
                    print("No tables or views found.")
                    return {}

            print(f"Found {len(views)} objects in schema '{schema_name}'. Extracting metadata and value samples...")
            os.makedirs(output_dir, exist_ok=True)

            with engine.connect() as conn:
                for view_name in views:
                    columns_raw = inspector.get_columns(view_name, schema=schema_name)
                    view_data = []

                    for col in columns_raw:
                        col_name = col['name']

                        # NEW: Exclusion check
                        if (view_name.lower(), col_name.lower()) in self._excluded_columns:
                            print(f"   - Skipping excluded column from database metadata: {view_name}.{col_name}")
                            continue

                        val_info = {}
                        
                        try:
                            query = text(f'SELECT DISTINCT TOP 100 "{col_name}" FROM "{schema_name}"."{view_name}"')
                            result = conn.execute(query).fetchall()
                            raw_values = [row[0] for row in result if row[0] is not None]
                            
                            if len(raw_values) <= 10:
                                val_info["unique_values"] = [
                                    str(v) if not isinstance(v, (int, float, bool)) else v for v in raw_values
                                ]
                            else:
                                val_info["example_values"] = [
                                    str(v) if not isinstance(v, (int, float, bool)) else v for v in raw_values[:2]
                                ]
                        except Exception as col_err:
                            val_info = {"error": "Could not fetch values"}
                            print(f"   ! Values skip for {view_name}.{col_name}: {col_err}")

                        column_entry = {
                            "name": col_name,
                            "type": str(col['type']),
                            "nullable": col['nullable'],
                            "default": str(col['default']) if col.get('default') else None,
                        }
                        column_entry.update(val_info)
                        view_data.append(column_entry)

                    if view_data: # Only add view if it has non-excluded columns
                        metadata_registry[view_name] = view_data
                        file_path = os.path.join(output_dir, f"{view_name}_column_meta.json")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(view_data, f, indent=4, ensure_ascii=False)
                        
                        print(f" - Saved: {file_path}")
                    else:
                        print(f" - Skipped saving empty metadata for view '{view_name}' (all columns excluded).")

            print(f"\nDatabase extraction complete! Files are saved in '{output_dir}'.")
            return metadata_registry

        except Exception as e:
            print(f"Database error: {e}")
            return {}
        finally:
            if self.engine:
                self.engine.dispose()
                self.engine = None

    def export_column_desc_template(self, metadata_registry: dict, csv_output_path: str):
        """
        Creates a CSV template with TABLE_NAME, Column_Name, and an empty Description column
        based on the extracted database metadata, excluding columns in the exclusion list.
        """
        if not metadata_registry:
            print("No metadata registry available to export to CSV.")
            return

        rows = []
        for table_name, columns in metadata_registry.items():
            for col in columns:
                col_name = col["name"]
                # NEW: Exclusion check
                if (table_name.lower(), col_name.lower()) in self._excluded_columns:
                    # print(f"   - Skipping excluded column from template: {table_name}.{col_name}") # Optional: detailed skip msg
                    continue
                
                rows.append({
                    "TABLE_NAME": table_name,
                    "Column_Name": col_name,
                    "Description": ""  # Left empty for users to fill in manually later
                })

        try:
            output_dir = os.path.dirname(csv_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Exporting to CSV using pandas
            df = pd.DataFrame(rows)
            df.to_csv(csv_output_path, index=False, encoding='utf-8')
            print(f"Successfully generated column description template: '{csv_output_path}'")
        except Exception as e:
            print(f"Error generating CSV template file: {e}")

    def get_embedding_function(self, ollama_model="embeddinggemma", hf_model="google/embeddinggemma-300m"):
        """Checks for Ollama running locally; falls back to Hugging Face."""
        ollama_url = "http://localhost:11434"
        try:
            response = requests.get(ollama_url, timeout=2)
            if response.status_code == 200:
                print(f"Ollama detected. Using local model: {ollama_model}")
                return embedding_functions.OllamaEmbeddingFunction(
                    url=f"{ollama_url}/api/embeddings",
                    model_name=ollama_model
                )
        except requests.exceptions.ConnectionError:
            print("Ollama not found. Falling back to Hugging Face API.")

        return embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_name=hf_model
        )

    def create_chroma_db_from_csv(self, csv_file_path: str, batch_size: int = 100):
        """
        Creates/populates ChromaDB from a schema CSV. 
        Works for 'column_desc.csv' (with descriptions) and 'IDColumns.csv' (without descriptions).
        Excludes columns specified in the exclusion list.
        """
        try:
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        except FileNotFoundError:
            print(f"Error: The file '{csv_file_path}' was not found.")
            return

        table_col = next((c for c in df.columns if c.upper() in ['TABLE_NAME', 'TABLE']), None)
        column_col = next((c for c in df.columns if c.upper() in ['COLUMN_NAME', 'COLUMN']), None)
        desc_col = next((c for c in df.columns if c.upper() in ['DESCRIPTION', 'DESC']), None)

        if not table_col or not column_col:
            print(f"Error: CSV must contain table and column name fields. Columns found: {list(df.columns)}")
            return

        print(f"Initializing ChromaDB client at '{self.chroma_db_path}'...")
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        embedding_model = self.get_embedding_function()
        collection = client.get_or_create_collection(name=self.collection_name, embedding_function=embedding_model)

        documents, metadatas, ids = [], [], []

        print("Processing CSV records...")
        for index, row in df.iterrows():
            view_name = str(row[table_col])
            column_name = str(row[column_col])

            # NEW: Exclusion check
            if (view_name.lower(), column_name.lower()) in self._excluded_columns:
                print(f"   - Skipping excluded column for ChromaDB from CSV: {view_name}.{column_name}")
                continue

            if desc_col and pd.notna(row[desc_col]):
                doc_text = str(row[desc_col])
            else:
                doc_text = f"Key / identifier column '{column_name}' in database table/view '{view_name}'."

            documents.append(doc_text)
            metadatas.append({
                'view_name': view_name,
                'view_column': column_name,
                'source': 'csv_import'
            })
            ids.append(f"csv_{view_name}_{column_name}_{index}")

        self._insert_batches(collection, documents, metadatas, ids, batch_size)

    def create_chroma_db_from_metadata(self, metadata_registry: dict, view_desc_dict: dict = None, batch_size: int = 100):
        """
        Populates ChromaDB directly from live database schema metadata.
        Uses structural metadata + optional View Descriptions from JSON to map values accurately.
        Excludes columns specified in the exclusion list.
        """
        if not metadata_registry:
            print("No database metadata to load into ChromaDB.")
            return

        print(f"Initializing ChromaDB client at '{self.chroma_db_path}'...")
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        embedding_model = self.get_embedding_function()
        collection = client.get_or_create_collection(name=self.collection_name, embedding_function=embedding_model)

        documents, metadatas, ids = [], [], []

        for view_name, columns in metadata_registry.items():
            view_desc = view_desc_dict.get(view_name, "") if view_desc_dict else ""
            view_context_str = f" Context/Description: {view_desc}" if view_desc else ""

            for idx, col in enumerate(columns):
                col_name = col['name']

                # NEW: Exclusion check
                if (view_name.lower(), col_name.lower()) in self._excluded_columns:
                    print(f"   - Skipping excluded column for ChromaDB from metadata: {view_name}.{col_name}")
                    continue

                col_type = col['type']
                nullable = "nullable" if col['nullable'] else "non-nullable"
                
                sample_vals = col.get('unique_values') or col.get('example_values') or []
                sample_str = ", ".join([f"'{v}'" for v in sample_vals]) if sample_vals else "none"

                doc_text = (
                    f"Column '{col_name}' in table '{view_name}'.{view_context_str} "
                    f"Data Type: {col_type} ({nullable}). "
                    f"Sample/Categorical Values: [{sample_str}]."
                )

                documents.append(doc_text)
                metadatas.append({
                    'view_name': view_name,
                    'view_column': col_name,
                    'source': 'live_db_metadata'
                })
                ids.append(f"db_{view_name}_{col_name}_{idx}")

        self._insert_batches(collection, documents, metadatas, ids, batch_size)

    def _insert_batches(self, collection, documents, metadatas, ids, batch_size):
        """Internal helper to insert records into ChromaDB in chunks."""
        if not documents:
            print("No documents to insert into ChromaDB.")
            return

        print(f"Writing {len(documents)} items in batches of {batch_size}...")
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size]
            )
        print(f"Success! Total documents in vector database: {collection.count()}")


# =====================================================================
# PIPELINE EXECUTION ENGINE
# =====================================================================
if __name__ == '__main__':
    # Define the path to your exclusion CSV
    EXCLUDED_CSV = '../docs/to_be_exluded.csv' 

    pipeline = NL2SQLDataPipeline(
        chroma_db_path='../data/schema_db',
        collection_name='schema_collection',
        excluded_csv_path=EXCLUDED_CSV # Pass the exclusion file path here
    )

    # -------------------------------------------------------------
    # Step 0: Convert View Description CSV to JSON 
    # -------------------------------------------------------------
    print("--- STEP 0: Converting View Description CSV to JSON ---")
    VIEWS_CSV = '../docs/view_descriptions.csv'
    VIEWS_JSON = '../data/view_descriptions.json'
    
    view_desc_dict = {}
    if os.path.exists(VIEWS_CSV):
        view_desc_dict = pipeline.convert_view_desc_csv_to_json(VIEWS_CSV, VIEWS_JSON)
    else:
        print(f"Skipping Step 0: '{VIEWS_CSV}' not found. Place your View/Description CSV here.")

    # -------------------------------------------------------------
    # Step 1: Direct Database Inspection (Extract rich metadata)
    # -------------------------------------------------------------
    print("\n--- STEP 1: Running SQL Database Inspection ---")
    extracted_metadata = pipeline.extract_database_metadata(
        schema_name='Report', 
        output_dir='../data/metadata'
    )

    # -------------------------------------------------------------
    # Step 1.5: Export Column Template CSV (NEW)
    # -------------------------------------------------------------
    print("\n--- STEP 1.5: Exporting Column Description Template CSV ---")
    COLUMNS_DESC_CSV = '../docs/column_desc.csv'
    # Generate the template if it doesn't already exist so the user can populate it
    if extracted_metadata:
        pipeline.export_column_desc_template(extracted_metadata, COLUMNS_DESC_CSV)

    # -------------------------------------------------------------
    # Step 2: Convert Column Description CSV to JSONL
    # -------------------------------------------------------------
    print("\n--- STEP 2: Converting Column Description CSV to JSONL ---")
    COLUMNS_DESC_JSONL = '../data/column_description.jsonl'

    if os.path.exists(COLUMNS_DESC_CSV):
        pipeline.convert_column_desc_csv_to_jsonl(COLUMNS_DESC_CSV, COLUMNS_DESC_JSONL)
    else:
        print(f"Skipping Step 2: '{COLUMNS_DESC_CSV}' not found.")

    # -------------------------------------------------------------
    # Step 3: Vector Search DB Setup (ChromaDB population)
    # -------------------------------------------------------------
    print("\n--- STEP 3: Building ChromaDB Search Index ---")

    # If the user has already populated the template CSV, build from it.
    # Otherwise, fallback to database metadata.
    if os.path.exists(COLUMNS_DESC_CSV):
        print(f"Populating from user's key file: '{COLUMNS_DESC_CSV}'")
        pipeline.create_chroma_db_from_csv(COLUMNS_DESC_CSV)
    else:
        print(f"'{COLUMNS_DESC_CSV}' not found.")
        print("Fallback: Dynamically indexing vector database from live DB metadata...")
        pipeline.create_chroma_db_from_metadata(
            metadata_registry=extracted_metadata, 
            view_desc_dict=view_desc_dict
        )

    # -------------------------------------------------------------
    # Step 4: Verify & Test
    # -------------------------------------------------------------
    print("\n--- STEP 4: Verification Query ---")
    client = chromadb.PersistentClient(path='../data/schema_db')
    try:
        col = client.get_collection(name='schema_collection')
        query_word = "payment id"
        print(f"Testing search for: '{query_word}'")
        results = col.query(query_texts=[query_word], n_results=3)
        
        if results and results['documents'] and results['documents'][0]:
            for i, text_doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                print(f"  Result {i+1}:")
                print(f"    Document: {text_doc}")
                print(f"    Metadata: {meta}")
        else:
            print("No results found for the query.")
    except Exception as e:
        print(f"Could not fetch tests: {e}")