import os
import shutil
import time
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Definições ---
DATA_PATH = 'data'
DB_PATH = 'db'
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

# --- 0. Inicializar Motor e Verificar Banco Existente ---
print("\n--- 0. Verificando Banco de Dados ---")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Conecta ao banco (se não existir, ele cria um vazio)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Puxa os metadados de tudo que já está salvo para descobrir os arquivos existentes
dados_existentes = vectorstore.get(include=['metadatas'])
arquivos_ja_indexados = set()

if dados_existentes and dados_existentes['metadatas']:
    for meta in dados_existentes['metadatas']:
        source = meta.get('source')
        if source:
            arquivos_ja_indexados.add(source)

print(f"Arquivos já mapeados no banco de dados: {len(arquivos_ja_indexados)}")

# --- 1. Carregar os Documentos da Pasta ---
print("\n--- 1. Lendo pasta de documentos ---")
todos_documentos = []

pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
todos_documentos.extend(pdf_loader.load())

docx_loader = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader, use_multithreading=True)
todos_documentos.extend(docx_loader.load())

if not todos_documentos:
    print("Erro: Nenhum documento encontrado na pasta 'data'.")
    exit()

# --- 2. Filtro Incremental (A Mágica Acontece Aqui) ---
print("\n--- 2. Filtrando Arquivos Novos ---")
documentos_novos = []

for doc in todos_documentos:
    source = doc.metadata.get('source')
    if source not in arquivos_ja_indexados:
        documentos_novos.append(doc)

if not documentos_novos:
    print("✅ Nenhum arquivo novo encontrado. O banco já está 100% atualizado!")
    print("-" * 40)
    exit()

# Conta quantos arquivos físicos novos existem (pois a lista tem os chunks/páginas)
fontes_novas = set([d.metadata.get('source') for d in documentos_novos])
print(f"Foram encontrados {len(fontes_novas)} arquivos NOVOS para indexar.")

# --- 3. Gerar Sumários Automáticos APENAS para os novos ---
print("\n--- 3. Lendo arquivos novos e gerando Sumários com IA ---")
text_por_arquivo = {}
for doc in documentos_novos:
    source = doc.metadata.get('source', 'Desconhecido')
    if source not in text_por_arquivo:
        text_por_arquivo[source] = ""
    
    if len(text_por_arquivo[source]) < 3000:
        text_por_arquivo[source] += doc.page_content.replace('\n', ' ') + " "

llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
prompt_sumario = ChatPromptTemplate.from_template(
    "Aja como um arquivista catalogador. Escreva um ÚNICO parágrafo contínuo descrevendo o tema central do texto abaixo. "
    "REGRAS ABSOLUTAS: "
    "1. É PROIBIDO usar listas, tópicos, marcadores ou quebras de linha. "
    "2. Não copie frases do texto. Explique com suas palavras do que o documento trata. "
    "3. Seja extremamente conciso (máximo de 3 linhas).\n\n"
    "Texto:\n{text}"
)
chain_sumario = prompt_sumario | llm | StrOutputParser()

resumos_gerados = {}
for source, texto_base in tqdm(text_por_arquivo.items(), desc="Resumindo Arquivos Novos", unit="arq"):
    try:
        resumo = chain_sumario.invoke({"text": texto_base})
        resumo_limpo = resumo.replace('\n', ' ').replace('- ', '').replace('*', '').strip()
        if len(resumo_limpo) > 280:
            resumo_limpo = resumo_limpo[:277] + "..."
        resumos_gerados[source] = resumo_limpo
    except Exception as e:
        resumos_gerados[source] = "Erro ao gerar resumo automático."

# --- 4. Dividir os Documentos e Injetar Metadados ---
print("\n--- 4. Processando Texto e Injetando Metadados ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(documentos_novos)

for split in splits:
    source = split.metadata.get('source', 'Desconhecido')
    split.metadata['resumo_ia'] = resumos_gerados.get(source, "Resumo indisponível")

print(f"Total de novos chunks criados: {len(splits)}")

# --- 5. Adicionar ao Banco Vetorial Existente ---
print("\n--- 5. Adicionando ao ChromaDB ---")
start_time = time.time() 

batch_size = 100 
for i in tqdm(range(0, len(splits), batch_size), desc="Gerando Embeddings", unit="lote"):
    batch = splits[i : i + batch_size]
    vectorstore.add_documents(batch) # Adiciona no fim, sem apagar o que existe

elapsed_time = time.time() - start_time 
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("-" * 40)
print(f"✅ Concluído! {len(fontes_novas)} novos arquivos adicionados com sucesso ao '{DB_PATH}'.")
print(f"Tempo total de indexação vetorial: {minutes}m {seconds}s ({elapsed_time:.2f} segundos).")
print("-" * 40)