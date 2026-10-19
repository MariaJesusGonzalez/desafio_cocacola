# Desafío técnico: Data Scientist Lead

Análisis de demanda y planificación de entregas para seis centros de
distribución, con datos del 1 de enero al 1 de marzo de 2026.

El proyecto evalúa el pronóstico base, analiza el plan de entrega, desarrolla
un modelo complementario Elastic Net e implementa una regla de historia
mínima de 45 días. Incluye una propuesta de arquitectura AWS y una
presentación ejecutiva de nueve diapositivas.

## Archivos principales

| Archivo o carpeta | Contenido |
| --- | --- |
| `desafio_data_scientist.ipynb` | Análisis de las cinco tareas, código y resultados. |
| `src/` | Funciones de carga, calidad de datos, métricas, modelo y resiliencia. |
| `data/` | Los cuatro CSV del desafío. |
| `outputs/` | Gráficos y arquitectura en formato editable `.drawio` y vista `.png`. |
| `presentation/presentacion_ejecutiva_v2.pdf` | Presentación ejecutiva final de nueve diapositivas. |
| `docs/` | Enunciado original del desafío. |
| `requirements.txt` | Versiones de las dependencias de Python. |
| `Dockerfile` y `DOCKER.md` | Contenedor e instrucciones de ejecución. |

## Ejecución local

Usar Python 3.12 y ejecutar los comandos desde la carpeta del proyecto.

```shell
python -m venv .venv
```

Activar el entorno con `.venv\Scripts\Activate.ps1` en PowerShell o
`source .venv/bin/activate` en Linux/macOS. Después:

```shell
python -m pip install -r requirements.txt
python -m jupyter lab
```

Abrir `desafio_data_scientist.ipynb` y ejecutar todas las celdas en orden.
Los resultados tabulares se muestran en el notebook; los dos gráficos se
guardan en `outputs/`. 

Para ejecutar en un contenedor, seguir [DOCKER.md](DOCKER.md).
