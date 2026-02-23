import sys
import time
# --- Importações de IA e Banco de Dados ---
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# --- Importações de Busca (Retrievers) ---
from langchain_classic.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever

# --- Importações do Core do LangChain ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Definições ---
DB_PATH = 'db'
MODEL_NAME = "deepseek-llm"      # Modelo para Embeddings
LLM_MODEL = "deepseek-llm"       # Modelo para o Chat

print("--- Inicializando o DeepArchive ---")

# 1. Carregar Embedding e Banco Vetorial
print("1. Carregando memória vetorial...")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# 2. Configurar Busca Híbrida (BM25 + Chroma)
print("2. Indexando palavras-chave (BM25) em memória...")
data = vectorstore.get()
doc_objects = [Document(page_content=c, metadata=m) for c, m in zip(data['documents'], data['metadatas'])]

if not doc_objects:
    print("ERRO: O banco de dados está vazio! Rode o 'index.py' primeiro.")
    sys.exit()

bm25_retriever = BM25Retriever.from_documents(doc_objects)
bm25_retriever.k = 5

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# 3. Configurar o Cérebro (LLM)
print(f"3. Conectando ao modelo de chat ({LLM_MODEL})...")
llm = ChatOllama(model=LLM_MODEL)

# 4. O Prompt
template = """Você é um assistente de pesquisa acadêmica chamado DeepArchive.
Use APENAS os contextos fornecidos abaixo para responder à pergunta do usuário.
Se a resposta não estiver nos contextos, diga que não sabe. Não invente informações.
Cite o nome dos arquivos fonte sempre que possível no corpo do texto.

Contextos:
{context}

Pergunta: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)

def clean_source_name(source_path):
    if "\\" in source_path: return source_path.split("\\")[-1]
    elif "/" in source_path: return source_path.split("/")[-1]
    return source_path

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Desconhecido')
        clean_name = clean_source_name(source)
        formatted_docs.append(f"[Fonte: {clean_name}]:\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

# 5. Criar a "Corrente" APENAS de Geração
generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# ==============================================================================
# SELEÇÃO DE MODO
# ==============================================================================
print("\n" + "="*50)
print("⚙️  ESCOLHA O MODO DE OPERAÇÃO")
print("1: Busca Rápida (Retorna apenas os trechos exatos dos documentos)")
print("2: Assistente RAG (A IA lê os documentos e formula uma resposta)")
print("="*50)

modo_escolhido = input("Digite 1 ou 2: ").strip()

if modo_escolhido not in ['1', '2']:
    print("Opção inválida. Iniciando no Modo 1 (Busca Rápida) por padrão.")
    modo_escolhido = '1'

print(f"\n--- Sistema Pronto! (Modo {modo_escolhido}). Digite 'sair' para encerrar ---")

# 6. Loop de Conversa Unificado
while True:
    query = input("\nVocê: ")
    if query.lower() in ['sair', 'exit', 'quit']:
        break
    
    if not query.strip():
        continue

    start_time = time.time()
    print("\nDeepArchive buscando fontes...", end="", flush=True)
    
    # --- Passo A: Recuperar Documentos (Comum aos dois modos) ---
    retrieved_docs = ensemble_retriever.invoke(query)
    
    # ---------------------------------------------------------
    # FLUXO 1: BUSCA SIMPLES
    # ---------------------------------------------------------
    if modo_escolhido == '1':
        print(f"\n\n--- 🔎 Encontrei {len(retrieved_docs)} documentos relevantes: ---\n")
        
        for i, doc in enumerate(retrieved_docs):
            raw_source = doc.metadata.get('source', 'Desconhecido')
            clean_name = clean_source_name(raw_source)
            
            print(f"[Resultado {i+1}] 📂 Arquivo: {clean_name}")
            print(f"📄 Trecho: \"{doc.page_content[:300].replace(chr(10), ' ')}...\"") 
            print("-" * 50)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"⏱️  Tempo de busca: {elapsed_time:.2f} segundos")

    # ---------------------------------------------------------
    # FLUXO 2: MODO RAG
    # ---------------------------------------------------------
    elif modo_escolhido == '2':
        unique_sources = set()
        for doc in retrieved_docs:
            raw_source = doc.metadata.get('source', 'Desconhecido')
            unique_sources.add(clean_source_name(raw_source))
        
        context_text = format_docs(retrieved_docs)
        print("\rDeepArchive gerando resposta... ", end="") 
        
        try:
            for chunk in generation_chain.stream({"context": context_text, "question": query}):
                print(chunk, end="", flush=True)
            print("\n")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:
                time_str = f"{elapsed_time:.2f} segundos"
            else:
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                time_str = f"{minutes}m {seconds}s"

            print("-" * 50)
            print(f"⏱️  Tempo total: {time_str}")
            print(f"📂 Fontes consultadas: {', '.join(unique_sources)}")
            print("-" * 50)

        except Exception as e:
            print(f"\nOcorreu um erro na geração: {e}")