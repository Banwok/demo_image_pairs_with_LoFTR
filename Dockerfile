FROM python:3.8 as builder
COPY requirements.txt .

RUN pip3 install --user -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
CMD [ "python3", "test.py"]
