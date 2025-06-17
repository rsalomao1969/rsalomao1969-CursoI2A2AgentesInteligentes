import os
import zipfile
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine

app = Flask(__name__)

UPLOAD_FOLDER = 'Uploads'
EXTRACTED_FOLDER = 'extracted'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FOLDER'] = EXTRACTED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

# Configure LlamaIndex with Ollama
Settings.llm = Ollama(model="mistral", request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name="mistral")  # Use local embeddings

def unzip_file(zip_path, extract_path):
    print(f"Descompactando {zip_path} para {extract_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Arquivos extraídos: {zip_ref.namelist()}")
    except Exception as e:
        print(f"Erro ao descompactar {zip_path}: {e}")
        raise

def load_csv_files(extract_path):
    cabecalho_df = None
    itens_df = None
    print(f"Procurando arquivos em: {extract_path}")
    for root, _, files in os.walk(extract_path):
        for file in files:
            print(f"Arquivo encontrado: {file}")
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            first_lines = [f.readline().strip() for _ in range(3)]
                            print(f"Primeiras 3 linhas de {file} (encoding {encoding}):\n" + "\n".join(first_lines[:3]))
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines='warn')
                        if 'cabecalho' in file.lower():
                            cabecalho_df = df
                            print(f"Carregado Cabecalho: {file} with encoding {encoding}, colunas: {list(df.columns)}, linhas: {len(df)}")
                        elif 'itens' in file.lower():
                            itens_df = df
                            print(f"Carregado Itens: {file} with encoding {encoding}, colunas: {list(df.columns)}, linhas: {len(df)}")
                        break
                    except Exception as e:
                        print(f"Erro ao ler {file} with encoding {encoding}: {str(e)}")
                        continue
                if not (cabecalho_df is not None and 'cabecalho' in file.lower()) and not (itens_df is not None and 'itens' in file.lower()):
                    print(f"Falha ao carregar {file} com todos os encodings tentados.")
    if cabecalho_df is None or itens_df is None:
        print("Erro: Um ou ambos os arquivos CSV não foram carregados.")
    return cabecalho_df, itens_df

def is_safe_question(question):
    unsafe_keywords = ['import', 'exec', 'eval', 'os', 'system']
    return not any(keyword in question.lower() for keyword in unsafe_keywords)

def sanitize_column_name(name):
    return name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('Ç', 'C').replace('Ã', 'A').replace('Õ', 'O')

@app.route('/')
def index():
    return render_template('inicio.html')

@app.route('/process', methods=['POST'])
def process():
    question = request.form['question']
    if not is_safe_question(question):
        return render_template('inicio.html', response="Erro: Pergunta contém termos não permitidos.")
    
    zip_file = request.files['zipfile']
    filename = secure_filename(zip_file.filename)
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    zip_file.save(zip_path)
    
    extract_path = os.path.join(app.config['EXTRACTED_FOLDER'], filename.split('.')[0])
    os.makedirs(extract_path, exist_ok=True)
    unzip_file(zip_path, extract_path)
    
    cabecalho_df, itens_df = load_csv_files(extract_path)
    
    if cabecalho_df is None or itens_df is None:
        return render_template('inicio.html', response="Erro: Não foi possível carregar os arquivos CSV. Verifique o console para detalhes.")
    
    try:
        # Sanitize column names for SQL compatibility
        cabecalho_df.columns = [sanitize_column_name(col) for col in cabecalho_df.columns]
        itens_df.columns = [sanitize_column_name(col) for col in itens_df.columns]
        
        # Create SQLite database with SQLAlchemy
        db_path = 'sqlite:///temp.db'
        engine = create_engine(db_path)
        
        # Load DataFrames into SQLite
        cabecalho_df.to_sql('cabecalho', engine, if_exists='replace', index=False)
        itens_df.to_sql('itens', engine, if_exists='replace', index=False)
        
        # Create SQL database for LlamaIndex
        sql_database = SQLDatabase(engine, include_tables=['cabecalho', 'itens'])
        
        # Define table schema with relationship
        table_schema = {
            "cabecalho": {
                "columns": cabecalho_df.columns.tolist(),
                "primary_key": "CHAVE_DE_ACESSO"
            },
            "itens": {
                "columns": itens_df.columns.tolist(),
                "primary_key": "CHAVE_DE_ACESSO",
                "foreign_key": "CHAVE_DE_ACESSO REFERENCES cabecalho(CHAVE_DE_ACESSO)"
            }
        }
        
        # Create query engine
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=['cabecalho', 'itens'],
            table_schema=table_schema,
            verbose=True
        )
        
        # Execute query
        response = query_engine.query(question)
        
        # Clean up
        engine.dispose()
        os.remove('temp.db')
        
        return render_template('inicio.html', response=str(response))
    
    except Exception as e:
        return render_template('inicio.html', response=f"Erro ao processar a pergunta: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
