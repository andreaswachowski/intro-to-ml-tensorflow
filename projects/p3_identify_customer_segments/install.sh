#!/bin/sh

# PACKAGES="numpy==1.12.1\
PACKAGES="numpy\
  pandas==0.23.3\
  scipy==1.2.1\
  black\
  seaborn==0.8.1\
  matplotlib\
  scikit-learn==0.19.2\
  jupyter \
  jupytext \
  jupyter-contrib-nbextensions \
  notebook"

 pip install $PACKAGES
