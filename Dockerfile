FROM python:3.11-slim

WORKDIR /opt

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache

COPY . .

EXPOSE 8080

ENV FLASK_APP=app.py

ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=8080"]
CMD ["--reload"]