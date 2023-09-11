FROM python:3.8-slim

RUN pip install --upgrade pip
RUN pip install pipenv


WORKDIR /opt
COPY ["Pipfile","Pipfile.lock","./"] 

RUN pipenv install --system --deploy

COPY . /opt/


EXPOSE 8000


ENTRYPOINT  uvicorn main:app --host=0.0.0.0 --port=8000
