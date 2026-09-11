# Revisión de reproducibilidad, 11/09/2026

Esta copia incorpora las correcciones de la auditoría. Los datos, protocolos, checkpoints y resultados históricos se mantienen byte por byte. Las nuevas comprobaciones se escriben en `results/reproduced/`. El PDF y las fuentes del manuscrito sí se han revisado.

## Forward: política histórica y corrección

La política congelada es `q = Delta / J`, donde J es la derivada del precio de entrega. Para neutralizar la Delta del valor actual del contrato debe utilizarse `q = exp(r*h) * Delta / J`, porque el forward liquida al final del intervalo. La función nueva `spot_delta_to_forward_units_pv` implementa esta segunda definición; los ejecutores históricos conservan la primera para poder reproducir sus cifras.

`python -B checks/forward_pv_diagnostic.py` comprueba la neutralización contra diferencias finitas del valor actual de un forward con precio de entrega fijo, la equivalencia a tipo cero y la contabilidad de las liquidaciones guardadas. Genera la comparación de RMSE con y sin el factor y el diagnóstico de precios y Deltas negativos. No recorta predicciones ni reentrena modelos.

## Recuperación de etapas intermedias

Se han recuperado cuatro fuentes de `backups/tfm_compute_v2/scripts` del repositorio suministrado: 05 (evaluación predictiva), 11 (bootstrap), 19 (errores por escenario) y 20 (fronteras de fixing). Sus bytes originales y hashes están en `archive/recovered_scripts/PROVENANCE.json`. La copia de desarrollo es evidencia de procedencia; no se afirma que su fecha de recuperación sea la fecha de ejecución original.

Ejecutar desde la raíz, con las dependencias del proyecto:

```powershell
python -B checks/reproduce_recovered.py
```

El ejecutor nuevo adapta únicamente rutas y destinos. Importa la función histórica de bootstrap y ejecuta los otros tres scripts con argumentos explícitos; todas las salidas nuevas quedan en `results/reproduced/recovered`. No se recomienda ejecutar directamente los scripts archivados con sus rutas predeterminadas antiguas.

La comprobación del 11/09/2026 reproduce exactamente, tras lectura numérica de CSV, las 19.994 réplicas bootstrap conservadas, las 18 filas de bandas, los 47 pronósticos expanding y las tablas de top 50, distancias y número de fixings. Los contrastes HAC coinciden dentro de tolerancia 1e-8 relativa/1e-10 absoluta. El archivo `verification.json` conserva los resultados. Esto cierra las lagunas concretas de generación señaladas en la auditoría. La búsqueda exploratoria de hiperparámetros completa sigue fuera del flujo de reproducción final, como indicaba el README original.

## Rutas y configuración histórica

`historical_path_map.json` relaciona las rutas antiguas del protocolo ML con las actuales por identidad SHA-256. Todas las referencias incluidas tienen destino localizado. Los protocolos y metadatos no se reescriben para cambiar sus rutas.

`results/model_selection/selected_shared_config.json` corresponde a la selección del benchmark GBM previo (cinco entradas, 99.969 parámetros). Al transferir la arquitectura a las siete entradas Log-OU, el modelo final tiene 100.225 parámetros. El peso diferencial final es 0,25; no debe confundirse con el del benchmark anterior. La regla de selección de arquitectura es un error estándar, no una desviación estándar.

## Entorno y fechas

Los metadatos de entrenamiento registran PyTorch 2.12.1+cu126 y CUDA 12.6. La revisión ha comprobado además el entorno virtual disponible del repositorio: Python 3.14.7, NumPy 2.5.2, pandas 2.3.3, SciPy 1.18.0 y statsmodels 0.14.6. Estas versiones se observaron el 11/09/2026 y se utilizaron para las comprobaciones de scripts recuperados; no son por sí solas una certificación retrospectiva de todas las dependencias del entrenamiento. `requirements.txt` expresa compatibilidad, no un lockfile histórico. No se ha reentrenado la cuadrícula ni regenerado el dataset RQMC completo.

La extracción Ornn H100 está fechada el 26/08/2026 en el CSV bruto; el panel multi-GPU, el 27/08/2026 en su manifiesto. Las muestras SMM corresponden al 26/08/2026. Las fechas de extracción se distinguen del periodo de observaciones y de las consultas bibliográficas.

## Compilación

Desde `thesis`, ejecutar `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`. La compilación utiliza XeLaTeX, Biber y las fuentes Times New Roman o TeX Gyre Termes. Se han corregido el vector desbordado, el idioma del abstract, los nombres de tablas y las anclas de página. Los avisos ambientales de MiKTeX sobre acceso a su carpeta de logs no son errores del manuscrito.

## Ajuste final a las normas de entrega

La portada lleva el número romano I y los preliminares continúan desde II; el cuerpo mantiene numeración arábiga desde 1. Se ha añadido al capítulo de datos el tratamiento de registros inválidos, duplicados, ausencias de calendario y valores extremos. La procedencia de los scripts recuperados se expresa como ubicación relativa dentro del repositorio de origen (no como dependencia de ejecución); se conservan sus hashes.
