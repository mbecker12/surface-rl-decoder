FROM python:3.8.5-slim as stage

RUN apt-get update && apt-get install -y \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    gcc \
    git

ENV WORKDIR_NAME=surface-rl-decoder
RUN mkdir ${WORKDIR_NAME}
WORKDIR /${WORKDIR_NAME}

ENV VIRTUAL_ENV=/${WORKDIR_NAME}/virtualenv/qec
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV VENV_PYTHON="${VIRTUAL_ENV}/bin/python"
ENV VENV_PIP="${VIRTUAL_ENV}/bin/pip"

COPY requirements.txt /${WORKDIR_NAME}/requirements.txt
RUN ${VENV_PYTHON} -m pip install --upgrade pip --no-cache-dir
RUN ${VENV_PIP} install -r requirements.txt --no-cache-dir

RUN mkdir /${WORKDIR_NAME}/runs
RUN chown quantum:quantum /${WORKDIR_NAME}/runs
COPY setup.cfg /${WORKDIR_NAME}/setup.cfg
COPY setup.py /${WORKDIR_NAME}/setup.py
COPY .git /${WORKDIR_NAME}/.git
COPY src /${WORKDIR_NAME}/src
COPY README.rst /${WORKDIR_NAME}/README.rst

RUN ${VENV_PYTHON} setup.py develop

RUN groupadd -r quantum && useradd -m -r -s /bin/sh -g quantum quantum
RUN chown quantum:quantum /${WORKDIR_NAME}
USER quantum

CMD /bin/bash
