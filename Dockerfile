FROM ubuntu:bionic

WORKDIR /workspace

COPY requirements.txt /workspace

RUN apt-get update && apt-get dist-upgrade -y && apt-get autoremove -y
RUN apt-get install python3 python3-pip -y

RUN pip3 install -r requirements.txt
RUN ln -s `which python3` /usr/bin/python

COPY . /workspace

ENTRYPOINT ["python3", "StarterScript.py"]
