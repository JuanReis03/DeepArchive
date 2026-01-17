import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Definições ---
DATA_PATH = 'data' # Pasta onde você "sobe" seus arquivos
DB_PATH = 'db'     # Pasta onde o banco de dados vetorial será salvo
MODEL_NAME = "deepseek-llm"

# --- 1. Carregar os Documentos (PDF e DOCX) ---
print("Carregando documentos...")

documents = []

# 1.1 Carregar PDFs
print("--- Buscando PDFs ---")
pdf_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)
pdf_docs = pdf_loader.load()
documents.extend(pdf_docs) # Adiciona os PDFs à lista principal

# 1.2 Carregar DOCX
print("--- Buscando Arquivos Word (.docx) ---")
docx_loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader, # Loader específico para Word
    show_progress=True,
    use_multithreading=True
)
docx_docs = docx_loader.load()
documents.extend(docx_docs) # Adiciona os DOCX à lista principal

# Verifica se carregou alguma coisa
if not documents:
    print("Nenhum documento (PDF ou DOCX) encontrado na pasta 'data'.")
    exit()

# Pegamos o nome dos arquivos únicos a partir dos metadados
unique_files = set(doc.metadata['source'] for doc in documents)

print(f"\nResumo: Carregados {len(documents)} páginas/documentos de {len(unique_files)} arquivos únicos.")

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

print(f"Modelo de embedding '{MODEL_NAME}' carregado.")

# --- 4. Criar e Salvar o Banco Vetorial ---
print("Iniciando a criação do banco de dados vetorial...")

# IMPORTANTE: Se o banco já existir, isso pode duplicar dados se você rodar várias vezes
# sem apagar a pasta 'db' antes. Para testes, é bom apagar a pasta 'db' manualmente
# antes de rodar este script novamente.
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"Banco de dados vetorial criado e salvo em '{DB_PATH}' com sucesso!")