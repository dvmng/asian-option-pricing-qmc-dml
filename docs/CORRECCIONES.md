# Correcciones del TFM — 11/09/2026

Se han aplicado las correcciones en el orden recomendado por la auditoría: definición financiera, interpretación de resultados, reproducibilidad y revisión editorial. El proyecto original permanece intacto; el ZIP contiene la copia revisada completa.

## Cambios principales

1. **Forward.** Se distingue el precio de entrega del valor actual del contrato y se incorpora el ratio correcto `q = exp(r*h) Delta/J`. La política histórica `Delta/J` y sus tablas permanecen identificadas. Una función nueva y un diagnóstico ejecutable comprueban la corrección; los RMSE posteriores son 0,002309264 para Oracle, 0,106781890 para MLP y 0,038794595 para DML. La jerarquía no cambia.
2. **Interpretación y muestras.** Se aclaran los estados interiores de la prueba local frente a las fronteras de la dinámica; la información completa del calendario contenida en τ; la igualdad modelizada de las leyes física y Q₀; y la ausencia de un factor de base independiente. Se incorporan los denominadores, la dependencia entre observaciones, el filtro de permanencia de 2.038/2.048 trayectorias y el alcance limitado de la dispersión entre semillas.
3. **Restricciones financieras.** Se documentan frecuencias y magnitudes de precios y Deltas negativos, tanto en test como en las decisiones dinámicas, sin recorte retrospectivo ni atribución causal de todas las colas.
4. **Reproducibilidad.** Se recuperan cuatro scripts de la copia de desarrollo, preservados con hashes y procedencia. Un ejecutor adapta las rutas y escribe resultados separados. Se añade un mapa de rutas históricas por identidad SHA-256, se identifica la configuración GBM previa y se distingue el entorno observado actualmente del acreditado históricamente por los metadatos.
5. **Descripción estadística.** Se corrige «un error estándar»; se explican medias, desviaciones, percentiles y máximos agregados por semilla; se precisan los controles numéricos de 12 estados y el uso de float32; se incorporan los resultados HAC y las seis réplicas bootstrap descartadas.
6. **Manuscrito.** Se corrigen redondeo y denominador porcentual, atribución de Kemna–Vorst y Turnbull–Wakeman, títulos bibliográficos en sentence case, fechas de extracción, nombres de tablas, decimales e idioma del abstract. Se elimina el desbordamiento del vector y las advertencias de fuentes y anclas duplicadas. Se reducen repeticiones y finales de capítulo innecesariamente vacíos.

## Comprobaciones realizadas

- Control del repositorio: cinco apartados PASS, incluida la verificación de hashes de los scripts recuperados.
- Reproducción exacta, al leer los CSV numéricamente, de 19.994 réplicas bootstrap, 18 filas de bandas, 47 pronósticos y las tablas de errores por fronteras y número de fixings. Los contrastes HAC coinciden dentro de la tolerancia documentada.
- Neutralización del forward contrastada con diferencias finitas del valor actual del contrato, equivalencia con la política histórica a tipo cero y conciliación de liquidaciones con los errores terminales conservados.
- Datos, modelos, protocolos, resultados históricos, figuras y tablas originales: identidad SHA-256 preservada. El manifiesto completo está en `docs/integrity_review.json` dentro del ZIP.
- Fuentes Python revisadas sintácticamente; referencias LaTeX resueltas; compilación y revisión visual del PDF final. Los informes ejecutables están en `results/reproduced/`.

No se reentrenaron redes ni se regeneró el dataset completo. Los diagnósticos nuevos se presentan como posteriores al experimento. La coincidencia del entorno virtual actual con varias versiones del documento de traspaso no demuestra por sí sola la configuración histórica completa; esta limitación queda expresamente documentada.

El argumento central se conserva: DML mejora ampliamente la Delta y la neutralización local; en la política dinámica supera a MLP, pero la cola impide superar a no cubrir en RMSE agregado.
