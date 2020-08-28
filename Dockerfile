FROM nvcr.io/nvidia/pytorch:20.08-py3

RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    libpq-dev \
    gcc \
    locales \
    curl

# install nodejs for the JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash - \
    && apt-get install -y nodejs build-essential

USER root
WORKDIR /deal_or_no_deal/

# copy files to container
COPY setup.py README.md requirements.txt ./
COPY deal_or_no_deal/_version.py ./meezer/

# install libraries
RUN \
    pip3 install -U pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 list > /python_requirements.txt

# enable progress bars in Jupyter Lab
RUN conda install -c conda-forge nodejs=14.4.0
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# copy the rest of the files to the container
COPY . .
