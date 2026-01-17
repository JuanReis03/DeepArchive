import os
import shutil
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Definições ---
DATA_PATH = 'data'
DB_PATH = 'db'
MODEL_NAME = "deepseek-llm"

# --- 0. Limpeza Automática do Banco Antigo ---
if os.path.exists(DB_PATH):
    print(f"Detectada pasta '{DB_PATH}' antiga. Apagando para recriar do zero.")
    shutil.rmtree(DB_PATH) # Apaga a pasta e tudo dentro dela
    print("Pasta antiga removida com sucesso.")
else:
    print(f"Nenhum banco antigo encontrado. Criando nova pasta '{DB_PATH}'...")

# --- 1. Carregar os Documentos (PDF e DOCX) ---
print("\n--- 1. Carregando Documentos ---")
documents = []

# Carregar PDFs
pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
pdf_docs = pdf_loader.load()
print(f"PDFs carregados: {len(pdf_docs)} páginas.")
documents.extend(pdf_docs)

# Carregar DOCX
docx_loader = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader, use_multithreading=True)
docx_docs = docx_loader.load()
print(f"DOCXs carregados: {len(docx_docs)} documentos.")
documents.extend(docx_docs)

if not documents:
    print("Erro: Nenhum documento encontrado.")
    exit()

# --- 2. Dividir os Documentos ---
print("\n--- 2. Processando Texto ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print(f"Total de chunks criados: {len(splits)}")

# --- 3. Carregar Modelo de Embedding ---
print("\n--- 3. Carregando Modelo de IA ---")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
print(f"Modelo '{MODEL_NAME}' pronto.")

# --- 4. Criar Banco Vetorial (Com Cronômetro) ---
print("\n--- 4. Indexando no ChromaDB ---")
print("Iniciando a criação do banco de dados vetorial. Aguarde...")

start_time = time.time() # Inicia o cronômetro

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=DB_PATH
)

end_time = time.time() # Para o cronômetro
elapsed_time = end_time - start_time # Calcula a diferença

# Formatação do tempo (minutos e segundos)
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("-" * 40)
print(f"Concluido! Banco de dados salvo em '{DB_PATH}'.")
print(f"Tempo total de indexação: {minutes}m {seconds}s ({elapsed_time:.2f} segundos).")
print("-" * 40)