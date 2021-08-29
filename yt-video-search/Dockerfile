FROM python:3.9.5-slim-buster

RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install ftfy regex tqdm matplotlib
RUN pip install git+https://github.com/openai/CLIP.git

RUN pip install Pillow opencv-python-headless streamlit youtube_dl

COPY tasks/*.py .
