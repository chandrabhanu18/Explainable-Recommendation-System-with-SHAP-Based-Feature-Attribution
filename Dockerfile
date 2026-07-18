FROM python:3.10-slim
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s CMD python -c "import requests, sys; sys.exit(0 if requests.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000"]
