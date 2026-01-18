import sys
import time  # Importante para contar o tempo
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
LLM_MODEL = "deepseek-llm"   # Modelo para o Chat

print("--- Inicializando o DeepArchive (Modo RAG com Metadados) ---")

# 1. Carregar Embedding e Banco Vetorial
print("1. Carregando memória vetorial...")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# 2. Configurar Busca Híbrida (BM25 + Chroma)
print("2. Indexando palavras-chave (BM25) em memória...")
data = vectorstore.get()
# Recria objetos Document para o BM25
doc_objects = [Document(page_content=c, metadata=m) for c, m in zip(data['documents'], data['metadatas'])]

if not doc_objects:
    print("ERRO: O banco de dados está vazio! Rode o 'index.py' primeiro.")
    sys.exit()

bm25_retriever = BM25Retriever.from_documents(doc_objects)
bm25_retriever.k = 5  # Top 5 por palavra-chave

chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Top 5 por semântica

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5] # 50% peso para cada método
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

# Função auxiliar para limpar nomes de arquivos
def clean_source_name(source_path):
    if "\\" in source_path: return source_path.split("\\")[-1]
    elif "/" in source_path: return source_path.split("/")[-1]
    return source_path

# Função para formatar os documentos em uma string única
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Desconhecido')
        clean_name = clean_source_name(source)
        formatted_docs.append(f"[Fonte: {clean_name}]:\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

# 5. Criar a "Corrente" APENAS de Geração (A recuperação faremos manualmente no loop)
generation_chain = (
    prompt
    | llm
    | StrOutputParser()
)

print("\n--- Sistema Pronto! Pergunte sobre seus documentos. Digite 'sair' para encerrar ---")


# 6. Loop de Conversa - RAG
while True:
    query = input("\nVocê: ")
    if query.lower() in ['sair', 'exit', 'quit']:
        break
    
    if not query.strip():
        continue

    # Inicia a contagem do tempo
    start_time = time.time()

    print("\nDeepArchive buscando fontes...", end="", flush=True)
    
    # --- Passo A: Recuperar Documentos (Manual) ---
    retrieved_docs = ensemble_retriever.invoke(query)
    
    # --- Passo B: Extrair Fontes ---
    unique_sources = set() # Usamos um set para não repetir nomes
    for doc in retrieved_docs:
        raw_source = doc.metadata.get('source', 'Desconhecido')
        unique_sources.add(clean_source_name(raw_source))
    
    # --- Passo C: Formatar Contexto ---
    context_text = format_docs(retrieved_docs)

    print("\rDeepArchive gerando resposta... ", end="") 
    
    # --- Passo D: Gerar Resposta ---
    try:
        # Passamos o contexto já formatado e a pergunta
        for chunk in generation_chain.stream({"context": context_text, "question": query}):
            print(chunk, end="", flush=True)
        print("\n")
        
        # --- Passo E: Exibir Estatísticas ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Formata o tempo (se for menos de 60s mostra segundos, se for mais, mostra min:seg)
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

           
# ==============================================================================
# 🔴 MODO RAG / CHAT (DESATIVADO) - Tire os '"""' abaixo para desativar a IA
# ==============================================================================
"""      
# 6. Loop de Conversa - Padrão
while True:
    query = input("\nVocê: ")
    if query.lower() in ['sair', 'exit', 'quit']:
        break
    
    if not query.strip():
        continue

    # Inicia a contagem do tempo
    start_time = time.time()

    print("\nDeepArchive buscando fontes...", end="", flush=True)
    
    # --- Passo A: Recuperar Documentos (MANTENHA ISSO) ---
    retrieved_docs = ensemble_retriever.invoke(query)
    
    # ==============================================================================
    # 🟢 MODO BUSCA SIMPLES (ATIVO) - Use isso para mostrar apenas os documentos
    # ==============================================================================
    print(f"\n\n--- 🔎 Encontrei {len(retrieved_docs)} documentos relevantes: ---\n")
    
    for i, doc in enumerate(retrieved_docs):
        # Limpa o nome do arquivo usando sua função auxiliar ou split direto
        raw_source = doc.metadata.get('source', 'Desconhecido')
        if "\\" in raw_source: clean_name = raw_source.split("\\")[-1]
        elif "/" in raw_source: clean_name = raw_source.split("/")[-1]
        else: clean_name = raw_source
        
        # Mostra o resultado
        print(f"[Resultado {i+1}] 📂 Arquivo: {clean_name}")
        # Mostra os primeiros 300 caracteres do conteúdo
        print(f"📄 Trecho: \"{doc.page_content[:300].replace(chr(10), ' ')}...\"") 
        print("-" * 50)
    
    # Exibir estatísticas de tempo apenas
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  Tempo de busca: {elapsed_time:.2f} segundos")
    
    """
