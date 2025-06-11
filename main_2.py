#Bibliotecas
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import rarfile
import tempfile
from pathlib import Path


#Para rodar esse código é necessário que usem o UnRAR/WinRAR no PATH do Windows.
#Os arquivo(s) rar devem estar no mesmo diretório do main.py
#Rodar em VENV para install bibliotecas


def carregar_df_do_rar(diretorio_script="."):
    print("Procurando por arquivos .rar...")
    caminho = Path(diretorio_script)
    arquivos_rar = list(caminho.glob('*.rar'))
    caminho_do_rar = None
    #Escolha do RAR
    if not arquivos_rar:
        print("Nenhum arquivo .rar encontrado no diretório.")
        return None
    elif len(arquivos_rar) == 1:
        caminho_do_rar = arquivos_rar[0]
        print(f"Apenas um arquivo .rar encontrado: '{caminho_do_rar.name}'.")
    else:
        print("Múltiplos arquivos .rar encontrados. Por favor, escolha um para processar:")
        for i, arquivo in enumerate(arquivos_rar):
            print(f"  [{i + 1}] {arquivo.name}")
        while True:
            try:
                resposta = input(f"Digite o número do arquivo .rar (1-{len(arquivos_rar)}): ")
                escolha_idx = int(resposta) - 1
                if 0 <= escolha_idx < len(arquivos_rar):
                    caminho_do_rar = arquivos_rar[escolha_idx]
                    break
                else:
                    print("Opção inválida. Por favor, escolha um número da lista.")
            except (ValueError, IndexError):
                print("Entrada inválida. Por favor, digite apenas um dos números mostrados.")
    #Escolha do CSV
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nExtraindo '{caminho_do_rar.name}'...")
        try:
            with rarfile.RarFile(caminho_do_rar) as rf:
                rf.extractall(path=temp_dir)
            print("Buscando arquivos .csv nos arquivos extraídos...")
            arquivos_csv = list(Path(temp_dir).glob('**/*.csv'))
            caminho_do_csv = None
            if not arquivos_csv:
                print("Nenhum arquivo .csv encontrado dentro do .rar.")
                return None
            elif len(arquivos_csv) == 1:
                caminho_do_csv = arquivos_csv[0]
                print(f"Apenas um CSV encontrado: '{caminho_do_csv.name}'. Carregando automaticamente.")
            else:
                print("Múltiplos arquivos CSV encontrados. Por favor, escolha um:")
                for i, arquivo in enumerate(arquivos_csv):
                    print(f"  [{i + 1}] {arquivo.name}")
                while True:
                    try:
                        resposta = input(f"Digite o número do arquivo CSV (1-{len(arquivos_csv)}): ")
                        escolha_idx = int(resposta) - 1
                        if 0 <= escolha_idx < len(arquivos_csv):
                            caminho_do_csv = arquivos_csv[escolha_idx]
                            break
                        else:
                            print("Opção inválida. Por favor, escolha um número da lista.")
                    except (ValueError, IndexError):
                        print("Entrada inválida. Por favor, digite apenas um dos números mostrados.")
            print(f"Carregando dados de '{caminho_do_csv.name}'...")
            df = pd.read_csv(caminho_do_csv)
            return df
        except rarfile.Error as e:
            print(f"Erro ao descompactar o arquivo RAR: {e}")
            return None
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")
            return None

df = carregar_df_do_rar()
if df is None:
    print("Processo interrompido devido à falha no carregamento dos dados.")
    exit()

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        metadata = row.to_dict()
        page_content = ", ".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
        document = Document(
            page_content=page_content,
            metadata=metadata,
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="NFs",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": len(df)}
)

model = OllamaLLM(model="Mistral")
template = """
Expert em finanças, Balanço e Análise fiscal, você analisa cada nota fiscal buscando primeiramente a
chave de acesso equivalente

Notas Fiscais a analisar: {NFs}
Pergunta a responder: {Pergunta}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    Pergunta = input("Faça sua pergunta (s para sair): ")
    print("\n\n")
    if Pergunta.lower() == "s":
        break
    NFs_docs = retriever.invoke(Pergunta)
#   print(f"Documentos retornados pelo retriever: {len(NFs_docs)}") --> debug caso o retriver não encontre todos os CSV's, aconteceu algumas vezes...
    NFs = [doc.metadata for doc in NFs_docs]
    NFs_str = "\n\n".join(
        f"NF {i+1}:\n" + "\n".join(f"{k}: {v}" for k, v in nf.items())
        for i, nf in enumerate(NFs)
    )
    result = chain.invoke({"NFs": NFs_str, "Pergunta": Pergunta})
    print(result)