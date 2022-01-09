FROM python:3.8

WORKDIR /home

COPY pyoxynet/pyoxynet/test/testing.py ./

RUN pip install pyoxynet

CMD [ "python", "./testing.py"]