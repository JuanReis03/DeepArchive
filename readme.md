# DeepArchive

**DeepArchive** é um sistema de busca semântica inteligente projetado para aprimorar instrumentos de busca em arquivos permanentes. Este projeto, parte da minha pesquisa científica do grupo LTI DIGITAL, utiliza modelos de linguagem de larga escala (LLMs) para ir além da busca tradicional por palavras-chave.

O objetivo é criar um "serviço de referência digital" capaz de compreender o **contexto** e a **intenção** por trás de uma consulta, retornando resultados semanticamente relevantes, mesmo que os termos exatos não estejam presentes no documento.

## Tecnologias Utilizadas

* **Python 3.12+**
* **Ollama:** Ferramenta para servir e gerenciar os modelos LLM localmente, otimizando o uso de hardware.
* **Modelos de IA:**
  * `deepseek-llm`: Modelo principal para raciocínio e geração de respostas (Assistente RAG).
  * `llama3.2:1b`: Modelo quantizado e leve, dedicado exclusivamente à geração rápida de sumários no acervo.
  * `nomic-embed-text`: Modelo otimizado para a geração dos *embeddings* (representações vetoriais) dos textos.
* **LangChain:** O framework principal para construir o *pipeline* de processamento e orquestrar a IA.
* **Streamlit:** Framework utilizado para a construção da interface web interativa.
* **ChromaDB:** Banco de dados vetorial de código aberto usado para armazenar e consultar os metadados e embeddings.
* **BM25 (Rank_BM25):** Algoritmo de recuperação esparsa para a execução da Busca Híbrida.
* **Pandas & tqdm:** Bibliotecas utilizadas para formatação de tabelas de dados analíticos e barras de progresso no terminal.
* **Docx2txt / PyPDF:** Processamento e extração de texto estruturado.

## Status Atual do Projeto

O projeto evoluiu de um simples buscador de terminal para um **Assistente Inteligente Web Completo** e autônomo. O pipeline central está consolidado com as seguintes funcionalidades:

* **Indexação Incremental e Inteligente (`index.py`):** O sistema mapeia os arquivos locais (`.pdf` e `.docx`) e processa exclusivamente os documentos inéditos, economizando poder de processamento. Durante essa ingestão, um LLM lê as primeiras páginas e gera sumários automatizados que são salvos como metadados permanentes.
* **Vitrine do Acervo:** A interface apresenta um catálogo interativo gerado automaticamente, permitindo ao usuário consultar os sumários descritivos de todos os documentos indexados antes mesmo de realizar uma pesquisa.
* **Busca Híbrida (Hybrid Search):** Através de um *Ensemble Retriever*, o sistema combina a precisão matemática de palavras-chave exatas (BM25) com a compreensão de contexto da busca vetorial (ChromaDB), garantindo altíssima precisão.
* **Análise e Exportação de Dados:** Os resultados da busca rápida são agrupados por arquivo e organizados em uma Tabela Analítica. O usuário possui opções para exportar os dados brutos em `.csv` ou baixar os relatórios de respostas redigidas pela IA em formato `.txt`.
* **Pipeline RAG Rigoroso (`app.py`):** O assistente generativo não apenas busca os trechos, mas formula respostas redigidas baseando-se **estritamente** no acervo. O sistema possui travas antifalucinação que o impedem de usar conhecimentos prévios e forçam a citação da bibliografia utilizada.

## 📋 Pré-requisitos

Antes de clonar e executar o sistema, certifique-se de ter as seguintes ferramentas instaladas no seu ambiente:

* **[Python](https://www.python.org/downloads/) (v3.10 ou superior):** Recomendamos a versão 3.12. *Nota para usuários Windows: durante a instalação, certifique-se de marcar a caixa "Add python.exe to PATH".*
* **[Git](https://git-scm.com/downloads):** Para o clone do repositório.
* **[Ollama](https://ollama.com/download):** Motor necessário para rodar os Modelos de Linguagem (LLMs) localmente.

**Modelos de IA Necessários:**
Após instalar o Ollama, abra o seu terminal e faça o download dos modelos base executando os seguintes comandos:
`ollama pull deepseek-llm` (Modelo RAG)
`ollama pull llama3.2:1b` (Modelo de apoio/sumarização)
`ollama pull nomic-embed-text` (Modelo de Embeddings)

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-REPOSITÓRIO]
    cd busca-de-arquivos-IC
    ```

2.  **Instale o Ollama e os Modelos Locais:**
    * Baixe e instale o [Ollama](https://ollama.com/) no seu sistema.
    * Faça o download dos modelos necessários executando os comandos abaixo no seu terminal:
        ```bash
        ollama pull deepseek-llm
        ollama pull llama3.2:1b
        ollama pull nomic-embed-text
        ```

3.  **Crie e ative o ambiente virtual (venv):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *(No Linux/Mac: `source venv/bin/activate`)*

4.  **Instale as dependências:**
    ```bash
    # Primeiro, PyTorch com suporte a CUDA (Placas NVIDIA)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Depois, as bibliotecas do projeto e da interface
    pip install -U langchain langchain-community pypdf langchain-chroma langchain-ollama streamlit pandas tqdm rank_bm25 docx2txt
    ```

5.  **Adicione seus arquivos:**
    * Coloque os arquivos `.pdf` ou `.docx` que deseja indexar dentro da pasta `/data`.

6.  **Execute a Indexação (Sempre que adicionar arquivos novos):**
    * Este script identificará os documentos novos, criará os sumários com IA e atualizará o banco `db`.
        ```bash
        python index.py
        ```

7.  **Inicie a Aplicação Web:**
    * Inicie o servidor do Streamlit para abrir a interface gráfica no seu navegador.
        ```bash
        streamlit run app.py
        ```

## Planos Futuros (Roadmap)

Para as próximas etapas do projeto, o foco será direcionado para o aprimoramento da infraestrutura local, da autonomia do usuário e do processamento de arquivos não padronizados. Planeja-se a implementação de um sistema de injeção via web, que permitirá o upload de novos documentos diretamente pela interface do navegador sem a necessidade de manipular diretórios do sistema operacional. O controle do usuário sobre a pesquisa será refinado com a adição de filtros avançados por metadados, possibilitando restringir a busca a um único arquivo, autor ou intervalo de tempo antes de acionar a inteligência artificial. Na camada de processamento de texto, o sistema será atualizado com tecnologia OCR (Reconhecimento Óptico de Caracteres) para extrair informações de PDFs baseados em imagens ou documentos digitalizados mais antigos. Por fim, visando um ambiente de produção escalável e seguro para o hardware, o projeto passará por um ajuste fino no gerenciamento de VRAM da placa de vídeo para otimizar o tempo de inferência e, em seguida, será inteiramente conteinerizado utilizando Docker, garantindo que o ecossistema possa ser implantado em qualquer máquina através de um único comando.
