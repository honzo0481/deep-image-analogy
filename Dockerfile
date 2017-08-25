FROM jupyter/tensorflow-notebook

RUN conda install -c conda-forge keras -y
