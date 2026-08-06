import streamlit as st
import time
import pandas as pd
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
MODEL_NAME = "deepseek-llm" 
EMBEDDING_MODEL = "nomic-embed-text" 

# --- 1. Configuração Inicial da Página ---
st.set_page_config(page_title="DeepArchive", page_icon="📚", layout="wide")

# --- 2. Funções Auxiliares ---
def clean_source_name(source_path):
    if "\\" in source_path: return source_path.split("\\")[-1]
    elif "/" in source_path: return source_path.split("/")[-1]
    return source_path

# --- 3. Inicialização do Motor ---
@st.cache_resource(show_spinner="Iniciando o Motor de Busca e IA. Aguarde...")
def initialize_engine():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    data = vectorstore.get()
    doc_objects = [Document(page_content=c, metadata=m) for c, m in zip(data['documents'], data['metadatas'])]
    
    if not doc_objects:
        return None, None, None
    
    # --- NOVIDADE FASE 2.5: Montando o Catálogo do Acervo ---
    catalogo_acervo = {}
    for doc in doc_objects:
        source = clean_source_name(doc.metadata.get('source', 'Desconhecido'))
        if source not in catalogo_acervo:
            # Puxa o resumo gerado pela IA no index.py. Se não existir (bancos antigos), mostra aviso.
            catalogo_acervo[source] = doc.metadata.get('resumo_ia', 'Sem resumo disponível.')
            
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 3
    
    chroma_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.45} 
    )
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.6, 0.4]
    )
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.0)
    
    template = """Você é um assistente de pesquisa acadêmica rigoroso chamado DeepArchive. Sua função é responder à [Pergunta] baseando-se EXCLUSIVAMENTE nos [Trechos dos Documentos] recuperados do acervo.

    REGRAS ABSOLUTAS E INQUEBRÁVEIS:
    1. Responda APENAS com base nas informações contidas nos [Trechos dos Documentos]. É ESTRITAMENTE PROIBIDO inventar ou fornecer links, URLs ou sites da internet.
    2. Se a resposta para a [Pergunta] não estiver explicitamente contida nos [Trechos dos Documentos], você é PROIBIDO de tentar deduzir ou alongar o assunto. Você DEVE responder EXATAMENTE com a seguinte frase e nada mais: "As informações solicitadas não constam nos documentos do acervo."
    3. JAMAIS use seu conhecimento prévio de mundo para completar, inventar ou justificar respostas.
    4. JAMAIS atribua falas, opiniões ou ações a pessoas, autores ou personagens que não estão explicitamente citados nos [Trechos dos Documentos].
    5. OBRIGATÓRIO: Responda SEMPRE em Português do Brasil (PT-BR).
    6. REGRA PARA LISTAGENS: Quando o usuário pedir para listar, citar ou mostrar documentos/arquivos, olhe a tag [Fonte: ...] no início de cada trecho. Você deve criar uma lista limpa apenas com os nomes dos arquivos.
       Formato OBRIGATÓRIO para listar documentos:
       - nome_do_arquivo_1.pdf
       - nome_do_arquivo_2.docx

    [Trechos dos Documentos]:
    {context}

    [Pergunta]: {question}

    Resposta em Português:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    generation_chain = prompt | llm | StrOutputParser()
    
    # Exportamos também o vectorstore puro para podermos extrair a matemática depois
    return ensemble_retriever, generation_chain, catalogo_acervo, vectorstore

# --- 4. Carrega o sistema ---
retriever, chain, catalogo_acervo, vectorstore = initialize_engine()

if not retriever:
    st.error("ERRO: O banco de dados está vazio! Rode o 'index.py' no terminal primeiro.")
    st.stop()

# --- 5. Interface Visual Principal ---
st.title("📚 DeepArchive: Assistente de Pesquisa")
st.markdown("Pesquise em seus acervos acadêmicos e documentos locais de forma inteligente.")

with st.sidebar:
    st.header("⚙️ Configurações")
    modo_selecionado = st.radio(
        "Modo de Operação:",
        ("1️⃣ Busca Rápida (Semântica + Palavras Chaves)", "2️⃣ Assistente RAG (IA Generativa)")
    )
    st.info("O Assistente RAG é mais lento, mas sintetiza a informação para você.")
    
    # --- Vitrine do Acervo na Sidebar ---
    st.markdown("---")
    st.header("🗂️ Acervo Indexado")
    
    # Exibe a quantidade total de documentos anexados
    if catalogo_acervo:
        st.caption(f"**{len(catalogo_acervo)} documentos anexados**")
    
    with st.expander("Ver documentos e sumários"):
        if catalogo_acervo:
            # Ordena alfabeticamente. Símbolos e números vão para o topo automaticamente!
            catalogo_ordenado = sorted(catalogo_acervo.items(), key=lambda x: x[0].lower())
            
            for arquivo, resumo in catalogo_ordenado:
                st.markdown(f"**📂 {arquivo}**")
                st.caption(f"_{resumo}_") # Exibe o resumo em itálico e menorzinho
                st.markdown("---")
        else:
            st.write("Nenhum documento encontrado.")

# --- 6. Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! O que você deseja pesquisar no acervo hoje?"}]

# Loop que recria a tela sempre que o usuário interage com algum botão
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Restaura a Tabela Analítica no histórico
        if "df" in message:
            st.dataframe(
                message["df"],
                column_config={
                    "Relevância": st.column_config.ProgressColumn("Relevância", format="%f%%", min_value=0, max_value=100),
                    "Arquivo": st.column_config.TextColumn("Arquivo", width="medium"),
                    "Pág.": st.column_config.TextColumn("Pág.", width="small"),
                    "Trecho Encontrado": st.column_config.TextColumn("Trecho do Documento", width="large")
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Restaura os botões de download no histórico
        if "download_data" in message:
            if message["download_type"] == "csv":
                st.download_button(
                    label="💾 Baixar Tabela (CSV)", 
                    data=message["download_data"], 
                    file_name=f"busca_{i}.csv", 
                    mime="text/csv", 
                    key=f"dl_csv_{i}"
                )
            elif message["download_type"] == "txt":
                st.download_button(
                    label="💾 Baixar Resposta (TXT)", 
                    data=message["download_data"], 
                    file_name=f"resposta_{i}.txt", 
                    mime="text/plain", 
                    key=f"dl_txt_{i}"
                )

# --- 7. Processamento da Pergunta ---
if pergunta := st.chat_input("Digite o tema, autor ou conceito que deseja buscar..."):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        start_time = time.time()
        msg_index = len(st.session_state.messages) # Usado para criar chaves únicas para os botões
        
        # ---------------------------------------------------------
        # FLUXO 1: BUSCA RÁPIDA (Com Matemática de Similaridade)
        # ---------------------------------------------------------
        if "1️⃣" in modo_selecionado:
            with st.spinner("Calculando similaridade semântica e buscando documentos..."):
                # Retorna os documentos E a nota de distância matemática
                docs_and_scores = vectorstore.similarity_search_with_score(pergunta, k=4)
            
            if not docs_and_scores:
                resposta = "Nenhum documento encontrado para esta pesquisa."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            else:
                tabela_dados = []
                agrupamento = {}
                
                # Prepara os dados calculando a porcentagem
                for doc, distancia in docs_and_scores:
                    # ChromaDB retorna distância. Invertemos para criar uma porcentagem de similaridade
                    similaridade = 1 / (1 + distancia)
                    porcentagem = round(similaridade * 100, 1)

                    source = clean_source_name(doc.metadata.get('source', 'Desconhecido'))
                    pagina = doc.metadata.get('page', 'N/A')
                    if isinstance(pagina, int): pagina = str(pagina + 1)
                    texto = doc.page_content.replace('\n', ' ')
                    
                    tabela_dados.append({"Relevância": porcentagem, "Arquivo": source, "Pág.": pagina, "Trecho Encontrado": texto})
                    
                    if source not in agrupamento:
                        agrupamento[source] = []
                    agrupamento[source].append({"pag": pagina, "texto": texto, "score": porcentagem})
                
                df = pd.DataFrame(tabela_dados)
                csv_data = df.to_csv(index=False).encode('utf-8')
                
                cabecalho = f"**🔎 Encontrei {len(docs_and_scores)} trechos relevantes em {len(agrupamento)} arquivo(s):**\n\n"
                st.markdown(cabecalho)
                
                # Desenha as caixas com a justificativa matemática
                for source, trechos in agrupamento.items():
                    with st.expander(f"📂 {source} ({len(trechos)} trecho(s) retornado(s))"):
                        for t in trechos:
                            st.markdown(f"🎯 *Retornado com **{t['score']}%** de similaridade matemática.*")
                            st.markdown(f"**Pág. {t['pag']}:** {t['texto'][:350]}...")
                            st.markdown("---")
                
                # Desenha a Tabela Analítica
                st.markdown("### 📊 Tabela Analítica")
                st.dataframe(
                    df,
                    column_config={
                        "Relevância": st.column_config.ProgressColumn("Relevância", format="%f%%", min_value=0, max_value=100),
                        "Arquivo": st.column_config.TextColumn("Arquivo", width="medium"),
                        "Pág.": st.column_config.TextColumn("Pág.", width="small"),
                        "Trecho Encontrado": st.column_config.TextColumn("Trecho do Documento", width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                elapsed = time.time() - start_time
                rodape = f"\n*⏱️ Tempo de busca: {elapsed:.2f} segundos*"
                st.caption(rodape)
                
                # Botão de Download do CSV
                st.download_button(
                    label="💾 Baixar Tabela (CSV)", 
                    data=csv_data, 
                    file_name=f"busca_{msg_index}.csv", 
                    mime="text/csv", 
                    key=f"dl_csv_{msg_index}"
                )
                
                # Salva os dados na memória para sobreviver ao recarregamento da página
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": cabecalho + rodape, 
                    "df": df,
                    "download_data": csv_data,
                    "download_type": "csv"
                })

        # ---------------------------------------------------------
        # FLUXO 2: ASSISTENTE RAG (Download TXT)
        # ---------------------------------------------------------
        else:
            with st.spinner("Buscando fontes e analisando conteúdo para redigir a resposta..."):
                docs = retriever.invoke(pergunta)
            
            if not docs:
                resposta = "Não encontrei informações nos documentos para responder a isso. Os termos não atingiram a similaridade mínima necessária."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            else:
                context_text = "\n\n".join([f"[Fonte: {clean_source_name(d.metadata.get('source', '?'))}]:\n{d.page_content}" for d in docs])
                
                try:
                    placeholder = st.empty()
                    texto_gerado = ""
                    
                    for chunk in chain.stream({"context": context_text, "question": pergunta}):
                        texto_gerado += chunk
                        placeholder.markdown(texto_gerado + "▌") 
                    
                    placeholder.markdown(texto_gerado)
                    
                    elapsed = time.time() - start_time
                    fontes_unicas = set([clean_source_name(d.metadata.get('source', '?')) for d in docs])
                    
                    rodape = f"\n\n---\n**⏱️ Tempo total:** {elapsed:.2f}s | **📂 Fontes Consultadas:** {', '.join(fontes_unicas)}"
                    st.caption(rodape)
                    
                    # Prepara o arquivo TXT para download
                    conteudo_txt = f"Pergunta: {pergunta}\n\nResposta do DeepArchive:\n{texto_gerado}\n\nFontes Consultadas: {', '.join(fontes_unicas)}\nTempo de busca: {elapsed:.2f}s"
                    
                    # Botão de Download do TXT
                    st.download_button(
                        label="💾 Baixar Resposta (TXT)", 
                        data=conteudo_txt, 
                        file_name=f"resposta_rag_{msg_index}.txt", 
                        mime="text/plain", 
                        key=f"dl_txt_{msg_index}"
                    )
                    
                    # Salva tudo no histórico
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": texto_gerado + rodape,
                        "download_data": conteudo_txt,
                        "download_type": "txt"
                    })
                    
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")