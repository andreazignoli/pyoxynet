FROM python:3.8

WORKDIR /home

RUN mkdir static && mkdir templates

COPY app.py .
COPY requirements.txt .
ADD static /home/static
ADD templates /home/templates

EXPOSE 9098/tcp

RUN pip install -r requirements.txt

CMD [ "python", "./app.py"]