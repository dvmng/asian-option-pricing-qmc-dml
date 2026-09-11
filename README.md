# Valoración y cobertura de opciones asiáticas

Código, datos y evidencia reproducible del Trabajo Fin de Máster del Máster en Finanzas Cuantitativas.

El proyecto estudia la valoración de opciones asiáticas aritméticas sobre capacidad de cómputo y la estimación de Delta mediante Randomized Quasi-Monte Carlo (RQMC), un perceptrón multicapa (MLP) y Differential Machine Learning (DML). La evaluación incluye análisis fuera de muestra, diagnóstico de errores y ejercicios de cobertura local y dinámica.

## Memoria del TFM

**[Leer o descargar el TFM completo (PDF)](thesis/main.pdf)**

Autor: David Martín Gómez. Este repositorio contiene la versión final del código y la evidencia reproducible que acompaña a la memoria. Las instrucciones de comprobación y reproducción se incluyen a continuación.

## Estructura del repositorio

```text
tfm_compute_thesis/
├── data/
│   ├── raw/                         datos de mercado preservados
│   ├── processed/                   datos derivados y normalizados
│   ├── frozen/                      datasets exactos usados en el TFM
│   ├── staging/                     nuevas descargas no congeladas
│   └── reproduced/                  datasets regenerados
├── models/
│   ├── final/                       40 checkpoints usados en el TFM
│   └── reproduced/                  nuevos entrenamientos
├── results/
│   ├── calibration/
│   ├── diagnostics/
│   ├── hedging/
│   ├── market/
│   ├── ml/
│   ├── model_selection/
│   ├── pricing/
│   └── reproduced/                  resultados de nuevas ejecuciones
├── protocols/                       configuración metodológica final
├── src/                             implementación cuantitativa
├── experiments/                     ocho etapas ejecutables
├── checks/                          controles estructurales y de regresión
├── figures/                         figuras utilizadas por el TFM
├── tables/                          tablas generadas para el TFM
├── thesis/                          fuentes LaTeX y PDF final
├── main.py
├── pyproject.toml
└── requirements.txt
```

Los directorios `data/frozen/`, `models/final/` y los resultados finales conservan la evidencia utilizada en el documento. Las nuevas ejecuciones se escriben en `data/reproduced/`, `models/reproduced/` y `results/reproduced/` para evitar cualquier sobrescritura accidental.

## Entorno

Se requiere Python 3.11 o superior. La evaluación final se realizó con PyTorch 2.12.1 y CUDA 12.6; una GPU CUDA es recomendable para regenerar el dataset RQMC y reentrenar la cuadrícula completa de modelos.

Instalación:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

Si se utiliza CUDA, la instalación de PyTorch debe corresponder a la versión CUDA disponible en el equipo.

## Comprobación de la entrega

Ejecutar desde la raíz del repositorio:

```powershell
python main.py --checks
python main.py --layout
python main.py --outputs
```

`--checks` valida la integridad del dataset congelado, la cuadrícula de 40 modelos finales, los protocolos de cobertura y los principales resultados utilizados por el TFM. `--outputs` regenera las figuras y tablas del documento a partir de la evidencia final almacenada, sin reentrenar modelos.

## Flujo experimental

```text
Datos de mercado preservados
        ↓
Análisis descriptivo y calibración Log-OU
        ↓
Generación del dataset RQMC de precio y Delta
        ↓
Entrenamiento MLP / DML
        ↓
Evaluación fuera de muestra
        ↓
Diagnóstico de errores
        ↓
Cobertura local
        ↓
Cobertura dinámica
```

Las etapas ejecutables son:

| Etapa | Script | Función |
|---:|---|---|
| 1 | `01_market_data.py` | Inspección de datos preservados y descarga opcional de nuevos datos Ornn |
| 2 | `02_market_analysis.py` | Análisis descriptivo multi-GPU |
| 3 | `03_model_calibration.py` | Calibración física del modelo Log-OU |
| 4 | `04_pricing_dataset.py` | Generación del dataset RQMC de precio y Delta |
| 5 | `05_ml_training.py` | Entrenamiento de MLP y DML |
| 6 | `06_ml_evaluation.py` | Evaluación final sobre test y referencia |
| 7 | `07_local_hedging.py` | Validación de cobertura local |
| 8 | `08_dynamic_hedging.py` | Stress test de cobertura dinámica |

La ayuda específica de cada etapa puede consultarse con, por ejemplo:

```powershell
python main.py --experiment 5 -- --help
```

## Reproducción del experimento final

La siguiente secuencia regenera el núcleo computacional sin modificar la evidencia original.

Generar nuevamente el dataset de valoración:

```powershell
python main.py --experiment 4 -- --preset tfm --device cuda --output data/reproduced/pricing_dataset
```

Reentrenar la cuadrícula completa de 40 modelos. Si `data/reproduced/ml_prepared/train_order.npy` no existe, se reconstruye automáticamente con la semilla fijada en el protocolo del proyecto y se comprueba contra el orden congelado:

```powershell
python main.py --experiment 5 -- --data-dir data/reproduced/pricing_dataset --prepared-dir data/reproduced/ml_prepared --output-dir models/reproduced --full-grid --device cuda
```

Evaluar los modelos reproducidos:

```powershell
python main.py --experiment 6 -- --data-dir data/reproduced/pricing_dataset --results-dir models/reproduced --output-dir results/reproduced/ml_evaluation --device cuda --save-predictions
```

Repetir los ejercicios de cobertura con los modelos reproducidos:

```powershell
python main.py --experiment 7 -- --models-dir models/reproduced --output-dir results/reproduced/hedging/local --device cuda
python main.py --experiment 8 -- --models-dir models/reproduced --output-dir results/reproduced/hedging/dynamic_forward --device cuda
```

Los CSV reproducidos pueden contrastarse con la evidencia final mediante `checks/regression_check.py`. Por ejemplo:

```powershell
python checks/regression_check.py results/ml/final_evaluation/aggregate_metrics.csv results/reproduced/ml_evaluation/aggregate_metrics.csv
```

## Datos de mercado

Los datos utilizados por el TFM se conservan en `data/raw/` y `data/processed/`. Esto permite reproducir el análisis a partir de las observaciones preservadas sin depender de que una fuente web siga devolviendo exactamente el mismo histórico.

La opción `--refresh-ornn` de la etapa 1 descarga una observación nueva a `data/staging/`; no sustituye ni modifica los datos empleados en el trabajo.

## Selección de modelos

La evidencia de la selección de hiperparámetros y de la ponderación diferencial se conserva en `results/model_selection/` y `data/frozen/model_selection_validation/`. El repositorio de entrega reproduce el experimento final a partir del protocolo metodológico congelado; no reejecuta la búsqueda exploratoria completa de hiperparámetros.

## Documento

El manuscrito se encuentra en `thesis/`. Las figuras y tablas finales se almacenan en `figures/` y `tables/` y se regeneran con:

```powershell
python main.py --outputs
```

Los datasets congelados, los checkpoints y los resultados finales deben permanecer sin modificaciones para conservar la evidencia del experimento. Esta edición revisa el PDF y sus fuentes y distingue la política forward histórica de la corrección del valor actual.

## Correcciones y comprobaciones posteriores

Consultar `docs/REPRODUCIBILIDAD.md` para la procedencia de los cuatro scripts recuperados, el mapa de rutas históricas y el entorno verificado. Para reproducir las etapas recuperadas y los diagnósticos nuevos:

```powershell
python -B checks/reproduce_recovered.py
python -B checks/forward_pv_diagnostic.py
```

Los resultados se guardan en `results/reproduced/`; no sustituyen la evidencia congelada. Las tablas principales del manuscrito mantienen los resultados históricos y la corrección de descuento se presenta por separado.
