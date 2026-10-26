# start with a light weight Python image
# This gives our container a Python runtime environment
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

# avoid caching the installed dependencies in the Docker image
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 allows the container to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
