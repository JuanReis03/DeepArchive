# DeepArchive

**DeepArchive** é um sistema de busca semântica inteligente projetado para aprimorar instrumentos de busca em arquivos permanentes. Este projeto, parte da minha pesquisa científica do grupo LTI DIGITAL, utiliza modelos de linguagem de larga escala (LLMs) para ir além da busca tradicional por palavras-chave.

O objetivo é criar um "serviço de referência digital" capaz de compreender o **contexto** e a **intenção** por trás de uma consulta, retornando resultados semanticamente relevantes, mesmo que os termos exatos não estejam presentes no documento.

## 💻 Tecnologias Utilizadas

* **Python 3.12+**
* **DeepSeek (via Ollama):** O modelo de LLM usado para gerar os *embeddings* (representações vetoriais) dos textos.
* **Ollama:** Ferramenta para servir e gerenciar os modelos LLM localmente.
* **LangChain:** O framework principal para construir o *pipeline* de processamento (carregar, dividir, indexar, consultar).
* **ChromaDB:** O banco de dados vetorial de código aberto usado para armazenar e consultar os *embeddings*.

## 🚀 Status Atual (Fase 1 - Concluída)

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

O *pipeline* atual é a fundação. Os próximos passos focam em expandir as funcionalidades, a usabilidade e a robustez do sistema, conforme delineado no documento da pesquisa:

* **Melhorias na Ingestão de Dados:**
    * Adicionar suporte a múltiplos formatos de arquivo (`.txt`, `.docx`, `.md`).
    * Implementar OCR para extrair texto de PDFs baseados em imagem (scans).

* **Melhorias na Interface (UX):**
    * **[Prototipagem Rápida]** Substituir o `query.py` por uma interface web usando **Streamlit**.
    * Agrupar resultados da busca por arquivo de origem, exibindo as páginas relevantes (ex: "Arquivo X: págs 2, 5, 10").

* **Melhorias na IA ("Cérebro"):**
    * **[RAG]** Fazer com que o sistema **responda** às perguntas usando os *chunks* como contexto (Retrieval-Augmented Generation), em vez de apenas mostrar os *chunks*.
    * Implementar **filtragem por metadados** (ex: buscar "IA Generativa" APENAS em documentos de 2024).

* **Nível de Produção (Deploy):**
    * **[Containerização]** Criar um `Dockerfile` e `docker-compose.yml` para empacotar a aplicação, facilitando o deploy.
    * Estabelecer um *pipeline* de avaliação para medir a qualidade (precisão e *recall*) das respostas do sistema.