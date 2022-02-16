FROM tingjhenjiang/jupyterhub-run_pyr:ubuntu20.04-cuda

# -- Layer: cluster-base

ARG shared_workspace=/opt/workspace

RUN mkdir -p ${shared_workspace}

ENV SHARED_WORKSPACE=${shared_workspace}
ENV NVIDIA_DISABLE_REQUIRE="1"

# -- Layer: JupyterHub-base

ARG NB_USERs="user1"
ARG NB_UID="1001"
ARG NB_GID="100"
ARG PYTHON_VERSION="3.9"

# Ref: https://github.com/jupyterhub/jupyterhub-the-hard-way/blob/HEAD/docs/installation-guide-hard.md
# https://hub.docker.com/r/jupyter/base-notebook/dockerfile
# https://hub.docker.com/r/rocker/rstudio/Dockerfile
# https://github.com/grst/rstudio-server-conda/blob/master/docker/init2.sh


RUN . /envvarset.sh && \
    sed -i 's|http://free.nchc.org.tw|http://archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update -y && \
    TZ="Asia/Taipei" DEBIAN_FRONTEND="noninteractive" apt-get install -y libomp-dev && \
    ${CONDA_PATH}/bin/conda update -n base -c defaults conda && \
    ${CONDA_PATH}/bin/conda install -y -c conda-forge -p ${CONDA_PATH}/envs/python pathos fsspec dask transformers faiss mlflow kaggle nltk && \
    ${CONDA_PATH}/envs/python/bin/pip3 install condor-tensorflow tensorflow-addons arff azureml-core azureml-mlflow --no-cache-dir && \
    ${CONDA_PATH}/bin/conda env remove -p ${CONDA_PATH}/envs/r -y && \
    ${CONDA_PATH}/bin/conda env remove -p ${CONDA_PATH}/envs/beakerx -y && \
    apt-get remove --purge rstudio-server -y && \
    rm -rf /var/lib/apt/lists*/ && \
    rm -Rf /tmp/* && \
    apt-get clean && \
    apt-get autoclean && \
    ${CONDA_PATH}/bin/conda clean -a -y

# -- Runtime

EXPOSE 8000
EXPOSE 8787
WORKDIR ${SHARED_WORKSPACE}
VOLUME ${SHARED_WORKSPACE}
#VOLUME ${shared_workspace}
RUN chgrp $NB_GID ${SHARED_WORKSPACE} -R && chmod 771 ${SHARED_WORKSPACE} -R
ENV FINAL_RUN_INIT_SCRIPT=$FINAL_RUN_INIT_SCRIPT
CMD ["sh","-c","${FINAL_RUN_INIT_SCRIPT}"]