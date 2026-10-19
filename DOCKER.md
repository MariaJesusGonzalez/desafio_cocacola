# Ejecución con Docker

El contenedor utiliza Python 3.12 y ejecuta el notebook completo con
Jupyter nbconvert. Incluye el código, los cuatro CSV y el diagrama AWS.

## Requisito

Docker instalado y en ejecución, configurado para contenedores Linux.
Ejecutar los comandos desde la carpeta principal del proyecto.

## Construcción

```shell
docker build -t desafio-cocacola .
```

`pywinpty` se instala únicamente en Windows.

## Ejecución

En PowerShell:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\outputs,target=/app/outputs" desafio-cocacola
```

En Linux o macOS:

```bash
docker run --rm --mount "type=bind,source=$(pwd)/outputs,target=/app/outputs" desafio-cocacola
```

La carpeta `outputs` debe contener `arquitectura_tarea_5.png` y su fuente
editable `arquitectura_tarea_5.drawio`.

## Resultados generados

- `outputs/desafio_data_scientist_ejecutado.ipynb`: copia ejecutada del notebook.
- `outputs/wape_by_horizon.png`: WAPE por centro y horizonte.
- `outputs/client_plan_bias.png`: distribución del Bias de volumen y fecha por cliente.

