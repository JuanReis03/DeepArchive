import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Definições ---
DATA_PATH = 'data' # Pasta onde você "sobe" seus arquivos
DB_PATH = 'db'     # Pasta onde o banco de dados vetorial será salvo
MODEL_NAME = "deepseek-llm"

# --- 1. Carregar os Documentos ---
# Usamos DirectoryLoader para ler todos os PDFs da pasta 'data'
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",    # Padrão para encontrar arquivos (todos os PDFs)
    loader_cls=PyPDFLoader, # Qual loader usar para esses arquivos
    show_progress=True,   # Mostrar barra de progresso
    use_multithreading=True # Acelerar carregamento
)

documents = loader.load()

if not documents:
    print("Nenhum documento encontrado na pasta 'data'.")
    exit()

# Pegamos o nome dos arquivos únicos a partir dos metadados de cada documento
unique_files = set(doc.metadata['source'] for doc in documents)

print(f"Carregados {len(documents)} páginas de {len(unique_files)} arquivos PDF.")

# --- 2. Dividir os Documentos ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

print(f"Documentos divididos em {len(splits)} chunks.")

# --- 3. Carregar Modelo de Embedding ---
embeddings = OllamaEmbeddings(
    model=MODEL_NAME
)

print("Modelo de embedding carregado.")

# --- 4. Criar e Salvar o Banco Vetorial ---
# Este comando CRIA o banco. 
# Ele vai processar todos os 'splits' com o modelo de embedding
# e salvar o resultado no diretório 'DB_PATH'.
print("Iniciando a criação do banco de dados vetorial...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=DB_PATH  # Onde salvar os dados
)

print(f"Banco de dados vetorial criado e salvo em '{DB_PATH}' com sucesso!")