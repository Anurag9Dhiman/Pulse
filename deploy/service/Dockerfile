# Build from the repository root:
#   docker build -f deploy/service/Dockerfile -t par-service .
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[service,llm,gemini]"

EXPOSE 8000
# One process only: task and robot-connection state are in-memory.
CMD ["sh", "-c", "python -m par.service.db.migrate && par serve --host 0.0.0.0 --port 8000"]
