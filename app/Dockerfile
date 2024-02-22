FROM python:3.8

WORKDIR /home

RUN mkdir static && mkdir templates && mkdir tf_lite_models

COPY app.py .
COPY requirements.txt .
ADD static /home/static
ADD templates /home/templates
ADD tf_lite_models /home/tf_lite_models

EXPOSE 9098/tcp

RUN pip install -r requirements.txt

CMD [ "python", "./app.py"]