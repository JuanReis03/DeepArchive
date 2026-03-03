import streamlit as st
import time
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Definições ---
DB_PATH = 'db'
MODEL_NAME = "deepseek-llm" # Esse continua sendo o cérebro que conversa
EMBEDDING_MODEL = "nomic-embed-text" # O novo cérebro de busca

# --- 1. Configuração Inicial da Página ---
st.set_page_config(page_title="DeepArchive", page_icon="📚", layout="wide")

DB_PATH = 'db'
MODEL_NAME = "deepseek-llm"

# --- 2. Funções Auxiliares ---
def clean_source_name(source_path):
    if "\\" in source_path: return source_path.split("\\")[-1]
    elif "/" in source_path: return source_path.split("/")[-1]
    return source_path

# --- 3. Inicialização do Motor (Cacheado na Memória) ---
# O @st.cache_resource impede que o LLM e o Banco sejam recarregados a cada clique
@st.cache_resource(show_spinner="Iniciando o Motor de Busca e IA. Aguarde...")
def initialize_engine():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    data = vectorstore.get()
    doc_objects = [Document(page_content=c, metadata=m) for c, m in zip(data['documents'], data['metadatas'])]
    
    if not doc_objects:
        return None, None
    
    # 1. Limite da busca por palavras-chave (BM25)    
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 3
    
    # 2. Limite da busca semântica (ChromaDB)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.0)
    
    template = """Você é um assistente de pesquisa acadêmica rigoroso chamado DeepArchive. Sua função é analisar os documentos do acervo e responder com base ESTRITAMENTE neles.

    REGRAS ABSOLUTAS E INQUEBRÁVEIS:
    1. Responda APENAS com base nas informações contidas nos [Contextos] fornecidos abaixo.
    2. Se a resposta para a [Pergunta] não estiver explicitamente contida nos [Contextos], você é PROIBIDO de tentar deduzir ou alongar o assunto. Você DEVE responder EXATAMENTE com a seguinte frase e nada mais: "As informações solicitadas não constam nos documentos do acervo."
    3. JAMAIS use seu conhecimento prévio de mundo para completar, inventar ou justificar respostas.
    4. JAMAIS atribua falas, opiniões ou ações a pessoas, autores ou personagens que não estão explicitamente citados nos [Contextos].

    Contextos:
    {context}

    Pergunta: {question}

    Resposta:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    generation_chain = prompt | llm | StrOutputParser()
    
    return ensemble_retriever, generation_chain

# --- 4. Carrega o sistema ---
retriever, chain = initialize_engine()

if not retriever:
    st.error("ERRO: O banco de dados está vazio! Rode o 'index.py' no terminal primeiro.")
    st.stop()

# --- 5. Interface Visual Principal ---
st.title("📚 DeepArchive: Assistente de Pesquisa")
st.markdown("Pesquise em seus acervos acadêmicos e documentos locais de forma inteligente.")

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # O botão de rádio com os seus textos exatos
    modo_selecionado = st.radio(
        "Modo de Operação:",
        ("1️⃣ Busca Rápida (Semântica + Palavras Chaves)", "2️⃣ Assistente RAG (IA Generativa)")
    )
    
    st.markdown("---")
    st.info("O Assistente RAG é mais lento, mas sintetiza a informação para você.")

# --- 6. Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! O que você deseja pesquisar no acervo hoje?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. Processamento da Pergunta ---
if pergunta := st.chat_input("Digite o tema, autor ou conceito que deseja buscar..."):
    # Exibe e salva pergunta
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Processa a resposta
    with st.chat_message("assistant"):
        start_time = time.time()
        
        # ---------------------------------------------------------
        # FLUXO 1: BUSCA RÁPIDA
        # ---------------------------------------------------------
        if "1️⃣" in modo_selecionado:
            with st.spinner("Buscando documentos relevantes..."):
                docs = retriever.invoke(pergunta)
            
            if not docs:
                resposta = "Nenhum documento encontrado para esta pesquisa."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            else:
                st.markdown(f"**🔎 Encontrei {len(docs)} trechos relevantes:**")
                
                # Montamos uma string para o histórico
                resposta_historico = f"**🔎 Encontrei {len(docs)} trechos relevantes:**\n\n"
                
                for i, doc in enumerate(docs):
                    source = clean_source_name(doc.metadata.get('source', 'Desconhecido'))
                    texto = doc.page_content[:600].replace('\n', ' ') + "..."
                    
                    # Cria caixinhas expansíveis super elegantes
                    with st.expander(f"Resultado {i+1} 📂 {source}"):
                        st.write(texto)
                        
                    resposta_historico += f"- **{source}**: {texto[:100]}...\n"
                
                elapsed = time.time() - start_time
                rodape = f"\n*⏱️ Tempo de busca: {elapsed:.2f} segundos*"
                st.caption(rodape)
                
                # Salva um resumo no histórico
                st.session_state.messages.append({"role": "assistant", "content": resposta_historico + rodape})

        # ---------------------------------------------------------
        # FLUXO 2: ASSISTENTE RAG
        # ---------------------------------------------------------
        else:
            with st.spinner("Buscando fontes e analisando conteúdo para redigir a resposta..."):
                docs = retriever.invoke(pergunta)
            
            if not docs:
                resposta = "Não encontrei informações nos documentos para responder a isso."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            else:
                # Formata contexto
                context_text = "\n\n".join([f"[Fonte: {clean_source_name(d.metadata.get('source', '?'))}]:\n{d.page_content}" for d in docs])
                
                # Prepara o espaço vazio onde a IA vai "digitar" ao vivo
                placeholder = st.empty()
                texto_gerado = ""
                
                try:
                    # Faz o streaming (digitação ao vivo)
                    for chunk in chain.stream({"context": context_text, "question": pergunta}):
                        texto_gerado += chunk
                        # Adiciona um "bloco" no final para imitar cursor de texto
                        placeholder.markdown(texto_gerado + "▌") 
                    
                    # Retira o cursor ao finalizar
                    placeholder.markdown(texto_gerado)
                    
                    # Extrai fontes e tempo
                    elapsed = time.time() - start_time
                    fontes_unicas = set([clean_source_name(d.metadata.get('source', '?')) for d in docs])
                    
                    rodape = f"\n\n---\n**⏱️ Tempo total:** {elapsed:.2f}s | **📂 Fontes Consultadas:** {', '.join(fontes_unicas)}"
                    st.caption(rodape)
                    
                    # Salva no histórico
                    st.session_state.messages.append({"role": "assistant", "content": texto_gerado + rodape})
                    
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")