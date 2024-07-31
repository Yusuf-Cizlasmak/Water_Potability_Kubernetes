FROM python:3.8-slim

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /opt/


WORKDIR /opt

EXPOSE 8080

ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0 --port:8080