# Entorno Linux reproducible para correr las pruebas y los experimentos de OrdenRed.
# Uso: ver la sección «Con Docker» del README.
FROM python:3.13.18-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Usuario sin privilegios: los archivos que se creen en el montaje no quedan como root.
# En Linux con otro uid: docker compose build --build-arg UID=$(id -u)
ARG UID=1000
RUN useradd --create-home --uid "$UID" ordenred
USER ordenred

CMD ["python", "-m", "unittest", "-v"]
