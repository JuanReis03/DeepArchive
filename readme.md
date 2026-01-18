# DeepArchive

**DeepArchive** é um sistema de busca semântica inteligente projetado para aprimorar instrumentos de busca em arquivos permanentes. Este projeto, parte da minha pesquisa científica do grupo LTI DIGITAL, utiliza modelos de linguagem de larga escala (LLMs) para ir além da busca tradicional por palavras-chave.

O objetivo é criar um "serviço de referência digital" capaz de compreender o **contexto** e a **intenção** por trás de uma consulta, retornando resultados semanticamente relevantes, mesmo que os termos exatos não estejam presentes no documento.

## 💻 Tecnologias Utilizadas

* **Python 3.12+**
* **DeepSeek (via Ollama):** O modelo de LLM usado para gerar os *embeddings* (representações vetoriais) dos textos.
* **Ollama:** Ferramenta para servir e gerenciar os modelos LLM localmente.
* **LangChain:** O framework principal para construir o *pipeline* de processamento (carregar, dividir, indexar, consultar).
* **BM25 (Rank_BM25):** Algoritmo para busca por palavras-chave (Sparse Retrieval).
* **Docx2txt / PyPDF:** Processamento e ingestão de arquivos.
* **ChromaDB:** O banco de dados vetorial de código aberto usado para armazenar e consultar os *embeddings*.

## Fase 1 - Concluída

O *pipeline* central do DeepArchive está 100% funcional. O que já foi implementado:

* **Indexação de Documentos (`index.py`):**
    * Carregamento automático de todos os arquivos `.pdf` da pasta `/data`.
    * Fragmentação (*chunking*) dos textos em segmentos otimizados.
    * Geração de *embeddings* usando o modelo **DeepSeek (`deepseek-llm`)** servido localmente via Ollama.
    * Persistência dos vetores em um banco de dados **ChromaDB** local (na pasta `/db`).

* **Consulta Semântica (`query.py`):**
    * Um script de console interativo que recebe perguntas do usuário.
    * Geração de *embedding* para a consulta usando o mesmo modelo DeepSeek (garantindo consistência).
    * Realização da busca por similaridade (k=3) no ChromaDB, retornando os *chunks* de texto mais relevantes.
    * Exibição dos resultados com o conteúdo e a fonte (nome do arquivo e página).

## Status Atual (Fase 2 - RAG & Interface Híbrida)

O projeto evoluiu de um simples buscador para um **Assistente Inteligente Completo**. As funcionalidades atuais incluem:

* **Ingestão Multiformato (`index.py`):**
    * Suporte para leitura e processamento de arquivos **.pdf** e **.docx** (Word).
    * Limpeza automática do banco de dados antigo antes da reindexação.
    * Monitoramento de tempo de processamento.

* **Busca Híbrida (Hybrid Search):**
    * Combina a precisão da busca por palavras-chave (**BM25**) com o entendimento contextual da busca vetorial (**ChromaDB**).
    * Utiliza *Ensemble Retriever* para garantir que termos técnicos exatos e conceitos abstratos sejam encontrados com igual eficiência.

* **Pipeline RAG (`app.py`):**
    * O sistema não apenas busca, mas **lê** os documentos e **responde** à pergunta do usuário.
    * Citação explícita das fontes consultadas ao final da resposta.

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-REPOSITÓRIO]
    cd busca-de-arquivos-IC
    ```

2.  **Instale o Ollama:**
    * Baixe e instale o [Ollama](https://ollama.com/) no seu sistema.
    * Puxe o modelo DeepSeek que será usado:
        ```bash
        ollama pull deepseek-llm
        ```

3.  **Crie e ative o ambiente virtual (venv):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *(No Linux/Mac: `source venv/bin/activate`)*

4.  **Instale as dependências:**
    ```bash
    # Primeiro, PyTorch com suporte a CUDA
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Depois, as bibliotecas do projeto
    pip install -U langchain langchain-community pypdf langchain-chroma langchain-ollama
    ```

5.  **Adicione seus arquivos:**
    * Coloque os arquivos `.pdf` que deseja indexar dentro da pasta `/data`.

6.  **Execute a Indexação (Apenas uma vez):**
    * Este script processará os arquivos da pasta `/data` e criará o banco `db`.
        ```bash
        python index.py
        ```

7.  **Execute a Consulta:**
    * Inicie o script de busca interativo.
        ```bash
        python query.py
        ```
## 🗺️ Planos Futuros (Roadmap)

Com a fundação do RAG e da interface estabelecida, os próximos passos focam em robustez e funcionalidades avançadas:

* **Melhorias na Ingestão de Dados:**
    * Implementar **OCR** para extrair texto de PDFs baseados em imagem (documentos digitalizados antigos).

* **Melhorias na Busca e IA:**
    * Implementar **filtragem por metadados** (ex: permitir que o usuário filtre a busca por ano ou autor antes de perguntar).
    * Refinamento dos *prompts* do sistema para diferentes perfis de resposta (ex: "Modo Resumo" vs "Modo Detalhado").
    
* **Melhorias na Interface (UX):**
    * **[Prototipagem Rápida]** Substituir o `query.py` por uma interface web usando **Streamlit**.
    * Agrupar resultados da busca por arquivo de origem, exibindo as páginas relevantes (ex: "Arquivo X: págs 2, 5, 10").

* **Nível de Produção (Deploy):**
    * **[Containerização]** Criar um `Dockerfile` e `docker-compose.yml` para empacotar a aplicação.
    * Estabelecer um *pipeline* de avaliação automatizada para medir a precisão das respostas geradas.
