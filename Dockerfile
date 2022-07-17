FROM continuumio/anaconda3
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-opencv
RUN pip install -U pip
RUN conda install -y -c conda-forge dlib
WORKDIR /app
COPY ["environment.yml", "environment.yml"]
RUN conda env create --file environment.yml -n face
