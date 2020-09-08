FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    libpq-dev \
    gcc \
    locales \
    curl

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

# copy the rest of the files to the container
COPY . .
