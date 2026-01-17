from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- Definições ---
DB_PATH = 'db'
MODEL_NAME = "deepseek-llm"

print("--- Iniciando Configuração da Busca Híbrida ---")

# 1. Carregar Modelo de Embedding
embeddings = OllamaEmbeddings(model=MODEL_NAME)
print("1. Modelo de embedding carregado.")

# 2. Carregar o Banco Vetorial (Chroma)
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
print("2. Banco de dados vetorial carregado.")

# --- O TRUQUE DA BUSCA HÍBRIDA ---
# Para usar o BM25 (palavras-chave), precisamos dos textos originais.
# Vamos puxar todos os documentos que já estão salvos no Chroma para criar o índice BM25.
print("3. Construindo índice de palavras-chave (BM25)...")

# Pega todos os dados do banco
data = vectorstore.get() 
docs_content = data['documents']
docs_metadatas = data['metadatas']

# Recria objetos "Document" para o BM25
doc_objects = []
for content, meta in zip(docs_content, docs_metadatas):
    doc_objects.append(Document(page_content=content, metadata=meta))

# Cria o buscador por Palavra-Chave (BM25)
bm25_retriever = BM25Retriever.from_documents(doc_objects)
bm25_retriever.k = 5  # Trazer 5 melhores por palavra-chave

# Cria o buscador Semântico (Chroma)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Cria o ENSEMBLE (Híbrido)
# weights=[0.5, 0.5] significa que damos 50% de peso para palavra-chave e 50% para semântica
# Se quiser forçar mais a palavra exata, mude para [0.7, 0.3]
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5] 
)

print("--- Sistema Pronto! ---")

# --- 3. Loop Interativo ---
while True:
    query = input("\nDigite sua pergunta (ou 'sair'): ")
    if query.lower() == 'sair':
        break
    
    if not query.strip():
        continue

    print(f"\nBuscando por: '{query}'...")

    # Agora usamos o ensemble_retriever.invoke em vez de vectorstore.similarity_search
    results = ensemble_retriever.invoke(query)

    print(f"\n--- Top Resultados Híbridos ---")
    
    if not results:
        print("Nada encontrado.")
        continue

    # Mostramos os resultados (limitado a 5 para não poluir)
    for i, doc in enumerate(results[:5]):
        source = doc.metadata.get('source', 'N/A')
        # Tenta limpar o caminho para mostrar só o nome do arquivo
        if "\\" in source:
            source = source.split("\\")[-1]
        elif "/" in source:
            source = source.split("/")[-1]
            
        print(f"\n[Resultado {i+1}] Fonte: {source}")
        print(f"Trecho: {doc.page_content[:300]}...") # Mostra só os primeiros 300 caracteres
        print("-" * 40)