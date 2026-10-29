# Run the app for a group on one machine or server:
#   docker build -t xrd-workup .
#   docker run -p 8501:8501 xrd-workup        then open http://<host>:8501
#   docker run -p 8501:8501 -e MP_API_KEY=... xrd-workup   one shared Materials Project key for all users (optional)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-mp.txt requirements-lebail.txt ./
# the Materials Project search is included; build with --build-arg WITH_MP=0 to leave it out
ARG WITH_MP=1
RUN if [ "$WITH_MP" = "1" ]; then pip install --no-cache-dir -r requirements-mp.txt; else pip install --no-cache-dir -r requirements.txt; fi
# the Le Bail tab (rietx) is included; build with --build-arg WITH_LEBAIL=0 to leave it out
ARG WITH_LEBAIL=1
RUN if [ "$WITH_LEBAIL" = "1" ]; then pip install --no-cache-dir -r requirements-lebail.txt; fi
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
