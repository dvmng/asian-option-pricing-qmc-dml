# TFM — Arithmetic Asian Option: MC/QMC + MLP/DML

Código de la **fase de datos y experimento numérico**.

## Diseño implementado

Contrato principal:

- Arithmetic Asian call discretamente monitorizada.
- `K = 1`.
- `T = 1` año.
- 12 fixings mensuales.
- GBM risk-neutral.
- `q = 0` en el núcleo del TFM.

Inputs del futuro surrogate:

1. `S_t_K = S_t / K`
2. `C_t = (1/N) sum_{fixing<=t} S_fixing/K`
3. `tau = T - t`
4. `r`
5. `sigma`

Targets:

- `price_K = V_t / K`
- `delta = dV/dS_t`

`S_t/K` y el average pasado **no se generan independientemente**. Ambos salen
de la misma trayectoria histórica GBM.

## Archivos

- `config.py`: parámetros y semillas.
- `asian_simulation.py`: GBM, estados, MC, RQMC-Sobol, precio, Delta pathwise y finite differences.
- `generate_datasets.py`: crea train/validation/test/reference.
- `validate_datasets.py`: consistencia financiera y anti-leakage.
- `benchmark_mc_qmc.py`: comparación homogénea MC vs scrambled Sobol.
- `hk_real_data.py`: datos reales opcionales de Hong Kong para calibración/contexto.

## Instalación

```bash
pip install -r requirements.txt
```

## 1. Ejecutar primero un piloto

```bash
python generate_datasets.py --preset pilot --split all
python validate_datasets.py --preset pilot --fd-check
```

El piloto comprueba que todo funciona antes de lanzar el experimento final.

## 2. Dataset final previsto

```bash
python generate_datasets.py --preset tfm --split train
python generate_datasets.py --preset tfm --split val
python generate_datasets.py --preset tfm --split test
python generate_datasets.py --preset tfm --split reference
python validate_datasets.py --preset tfm --fd-check
```

El preset `tfm` es deliberadamente costoso. No lo ejecutes hasta verificar tiempos
con el piloto.

## 3. MC vs QMC

Piloto:

```bash
python benchmark_mc_qmc.py --preset pilot
```

Experimento final:

```bash
python benchmark_mc_qmc.py --preset tfm
```

La distinción metodológica es explícita:

- Sobol exterior en `generate_states`: diseño del espacio de parámetros.
- RQMC-Sobol interior en `price_state`: innovaciones Brownianas de las trayectorias.

Solo el segundo caso se denomina **QMC pricing**.

## 4. Datos reales de Hong Kong

Ejemplo con Tencent:

```bash
python hk_real_data.py --ticker 0700.HK --start 2021-01-01 --end 2026-08-21
```

Otros tickers útiles:

- `1810.HK`: Xiaomi
- `9988.HK`: Alibaba
- `0005.HK`: HSBC Holdings
- `1299.HK`: AIA
- `^HSI`: Hang Seng Index

El script usa:

- Yahoo Finance mediante `yfinance` para series diarias gratuitas del subyacente.
- API oficial de HKMA para HIBOR.

Los datos reales son **complementarios**. El núcleo del dataset de pricing sigue
siendo simulado; no se necesitan cotizaciones históricas de Asian options.
