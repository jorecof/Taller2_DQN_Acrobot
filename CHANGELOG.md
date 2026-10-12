# Registro de cambios

## Versión 2.0 — 12 de octubre de 2026

Esta versión complementa la entrega original del Taller 2 con más análisis de los resultados que ya estaban en el repositorio. No se volvió a ejecutar ningún entrenamiento ni se agregaron experimentos: el código y las salidas del notebook son los mismos de la versión 1.0.

### Análisis agregado

- Tabla de resultados principales (política aleatoria frente a DQN) al inicio del README y del notebook.
- Curva de aprendizaje dividida en cuatro fases, con número de episodios, retorno promedio y porcentaje de episodios resueltos en cada una, a partir de `dqn_learning_curve.csv`.
- Momento en que el agente alcanza el umbral de -100: timestep 26.991, después de 167 episodios.
- Mediana, rango y número de episodios largos en la evaluación final, a partir de `dqn_evaluation.csv`.
- Intervalo de confianza del 95 % para el promedio entre semillas (-88,9 a -71,0) y análisis conjunto de los 90 episodios evaluados, a partir de `results/seed_comparison/`.
- Diferencia entre el retorno de entrenamiento y el de evaluación (unos 7 pasos por episodio) como costo de mantener epsilon = 0,1.
- Observación sobre el mejor episodio de cada semilla (entre -62 y -68) y su relación con el mínimo físico del problema.
- Observaciones cortas después de cada resultado del notebook y una sección final de conclusiones.

### Corregido

- Cifras de la evaluación final que habían quedado de una ejecución anterior. Pasan de -77,60 ± 10,92 a -83,30 ± 14,03, de 78,60 a 84,30 pasos, la mejora de 422,40 a 416,70 puntos y la reducción de 84,28 % a 83,14 %. Afecta al README y a las secciones 12 y 13 del notebook.
- `results/evaluation_summary.csv` y `results/baseline_vs_dqn.csv`, que no coincidían con `dqn_evaluation.csv`.
- Hitos de la curva de aprendizaje en el README, que correspondían a otra corrida.
- Conclusión del README, que mencionaba una semilla con menor estabilidad cuando las tres resolvieron todos los episodios.
- Limitación de la sección 12 del notebook, que decía que solo se había usado una semilla.
- Referencia al libro de Lapan, actualizada a la 3.ª edición (2024).

### Actualizado

- Referencias: se agregaron el artículo original de DQN (2013) y el libro de Sutton y Barto.
- Reflexión sobre dificultades, con lo observado en la variación entre semillas y en la revisión de cifras.
- Limitaciones y conclusiones ajustadas al alcance del taller (política aleatoria frente a DQN).

### Sin cambios

- Código y salidas de `Taller2_DQN_Acrobot.ipynb`.
- `seed_comparison.py`, `run_config.json`, `requirements.txt` y `.gitignore`.
- Resultados de `results/seed_comparison/`, `dqn_evaluation.csv`, `dqn_learning_curve.csv` y todas las figuras.

## Versión 1.0 — 11 de octubre de 2026

Entrega original del Taller 2: DQN con Stable-Baselines3 sobre `Acrobot-v1`, baseline aleatorio, evaluación de 30 episodios y comparación entre las semillas 42, 25 y 123.
