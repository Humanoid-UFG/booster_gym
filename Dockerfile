# --- Estágio 1: Imagem Base e Dependências Essenciais ---

# Usamos a imagem oficial da NVIDIA com CUDA 11.8 e CUDNN 8 como base.
# A tag "-devel" inclui ferramentas de compilação necessárias para as bibliotecas Python.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Define o frontend como não interativo para evitar prompts durante o build
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema: Python 3.8, pip, git e bibliotecas gráficas
# Usamos o PPA deadsnakes para garantir a disponibilidade do Python 3.8.
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3-pip \
        git \
        vim \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        patchelf && \
    # Limpa o cache do apt para reduzir o tamanho da imagem
    rm -rf /var/lib/apt/lists/*

# Define python3.8 como o comando padrão para 'python3'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

# Atualiza o pip para a versão mais recente
RUN python3 -m pip install --no-cache-dir --upgrade pip

# --- Estágio 2: Ambiente Python e Bibliotecas de RL ---

# Define o diretório de trabalho padrão dentro do contêiner
WORKDIR /app

# Define a variável de ambiente LD_LIBRARY_PATH, crucial para o Isaac Gym
# encontrar as bibliotecas compartilhadas do Python.
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/python3.8/config-3.8-x86_64-linux-gnu

# Instala PyTorch 2.0 para CUDA 11.8 e a versão específica do NumPy.
# Fazemos isso em um passo separado para aproveitar o cache de camadas do Docker.
RUN pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    "numpy<2.0"

# --- Estágio 3: Instalação do Isaac Gym ---

# Define um argumento para o nome do arquivo do Isaac Gym.
# Isso permite flexibilidade se o nome do arquivo mudar no futuro.
ARG ISAACGYM_FILE=IsaacGym_Preview_4_Package.tar.gz

# Copia o arquivo do Isaac Gym (que deve estar no contexto do build) para a imagem
COPY ${ISAACGYM_FILE} /tmp/

# Descompacta o arquivo, instala a biblioteca Python e depois limpa os arquivos temporários
RUN tar -xvf /tmp/${ISAACGYM_FILE} -C /tmp && \
    pip install --no-cache-dir /tmp/isaacgym/python && \
    rm -rf /tmp/isaacgym /tmp/${ISAACGYM_FILE}

# --- Estágio 4: Instalação da Aplicação ---

# Copia o arquivo de requerimentos primeiro para otimizar o cache de camadas
COPY requirements.txt .

# Instala o restante das dependências Python do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código fonte do projeto para o diretório de trabalho no contêiner
COPY . .

# --- Estágio 5: Comando de Execução ---

# Define o comando padrão para iniciar um terminal bash interativo
# Isso permite que você execute qualquer script (train, play, etc.) manualmente.
CMD ["bash"]