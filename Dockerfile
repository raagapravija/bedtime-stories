# Bedtime Story Maker - web app in a container.
#   docker build -t bedtime-stories .
#   docker run --rm -p 8000:8000 --env-file .env bedtime-stories
# then open http://localhost:8000. The OpenAI key comes from --env-file at run
# time; it is never copied into the image (.dockerignore excludes .env).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
