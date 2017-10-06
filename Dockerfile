FROM jupyter/tensorflow-notebook

RUN conda install -c conda-forge pytest keras -y
