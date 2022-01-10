FROM python:3.8

WORKDIR /home

COPY app.py .
COPY requirements.txt .

EXPOSE 9099/tcp

RUN pip install -r requirements.txt

CMD [ "python", "./app.py"]