FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=module://matplotlib_inline.backend_inline

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY desafio_data_scientist.ipynb ./
COPY outputs/arquitectura_tarea_5.drawio ./outputs/arquitectura_tarea_5.drawio
COPY outputs/arquitectura_tarea_5.png ./outputs/arquitectura_tarea_5.png


CMD ["python", "-m", "nbconvert", "--to", "notebook", "--execute", "desafio_data_scientist.ipynb", "--output", "desafio_data_scientist_ejecutado.ipynb", "--output-dir", "outputs", "--ExecutePreprocessor.timeout=600", "--ExecutePreprocessor.kernel_name=python3"]
