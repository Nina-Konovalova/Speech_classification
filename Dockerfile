FROM python:3.8

RUN apt update
RUN apt install -y git
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean

RUN pip3 install --upgrade pip==21.1.3
RUN pip install --upgrade setuptools

RUN mkdir -p /usr/src/app/
RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/
COPY . /usr/src/app/

COPY requirements.txt /usr/src/app/

Run apt-get install -yq libsndfile1

RUN pip install -r requirements.txt
RUN pip install --upgrade pip

RUN pip install speechbrain
RUN pip install transformers
RUN pip install speechbrain transformers

EXPOSE 8081

CMD ["python", "app.py"]

