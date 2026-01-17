from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Definições (DEVEM SER AS MESMAS DO index.py) ---
DB_PATH = 'db'     # Pasta onde o banco de dados vetorial foi salvo
MODEL_NAME = "deepseek-llm"

# --- 1. Carregar Modelo de Embedding ---
# Precisamos do *mesmo* modelo para 'embutir' a pergunta
embeddings = OllamaEmbeddings(
    model=MODEL_NAME
)

print("Modelo de embedding carregado.")

# --- 2. Carregar o Banco Vetorial Existente ---
# Esta é a diferença: não usamos 'from_documents',
# apenas carregamos o banco que já existe no disco.
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

print("Banco de dados vetorial carregado.")

# --- 3. Loop Interativo de Perguntas ---
while True:
    query = input("\nDigite sua pergunta (ou 'sair' para terminar): ")
    if query.lower() == 'sair':
        break
    
    if not query.strip():
        continue

    # Realiza a busca por similaridade
    # k=5 -> traga os 5 chunks mais relevantes
    search_results = vectorstore.similarity_search(query, k=5)

    print("\n--- Resultados da Busca ---")
    if not search_results:
        print("Nenhum resultado encontrado.")
        continue

    for i, doc in enumerate(search_results):
        print(f"Resultado {i+1}:\n{doc.page_content}\n")
        
        # Tenta pegar a fonte (nome do arquivo) e a página
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')
        
        print(f"(Fonte: {source} | Página: {page})\n")