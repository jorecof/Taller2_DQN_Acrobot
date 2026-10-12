# Registro de cambios

## Versión 2.0 — 12 de octubre de 2026

Esta versión complementa la entrega original del Taller 2. No se volvió a ejecutar ningún entrenamiento: el código y las salidas del notebook principal son los mismos de la versión 1.0. Los cambios son de análisis y documentación, más un experimento complementario en un notebook aparte.

### Nuevo

- `Taller2_DQN_Acrobot_v2.ipynb`: implementación propia de DQN en PyTorch, basada en los capítulos 6 y 8 del libro de Lapan, con comparación de cuatro agentes (SB3 DQN, DQN propio, Double DQN y Dueling DQN) en las semillas 42, 25 y 123.
- `results/variantes/`: figuras de curvas de aprendizaje, comparación de agentes y epsilon/pérdida/Q medio, y el archivo `resumen_resultados.md` con las tablas por semilla.
- Tabla de resultados principales al inicio del README y del notebook.
- Sección 5.6 del README y sección 14 del notebook con el experimento de variantes de DQN.
- Sección 15 de conclusiones en el notebook.
- Este archivo `CHANGELOG.md`.

### Análisis agregado

- Curva de aprendizaje dividida en cuatro fases, con número de episodios, retorno promedio y tasa de éxito de cada una, calculados desde `dqn_learning_curve.csv`.
- Momento en que el agente alcanza el umbral de -100: timestep 26.991, después de 167 episodios.
- Mediana, rango y número de episodios largos en la evaluación final.
- Intervalo de confianza del 95 % para el promedio entre semillas (-88,9 a -71,0) y análisis conjunto de los 90 episodios evaluados.
- Diferencia entre el retorno de entrenamiento y el de evaluación (unos 7 pasos por episodio) como costo de mantener epsilon = 0,1.
- Observación sobre el mejor episodio, que se repite en -62/-63 en todos los agentes y parece cercano al mínimo físico del problema.
- Comparación de variantes: Dueling DQN fue el mejor (-73,08); Double DQN no mejoró y no se observó sobreestimación de los valores Q.

### Corregido

- Cifras de la evaluación final que habían quedado de una ejecución anterior. Pasan de -77,60 ± 10,92 a -83,30 ± 14,03, de 78,60 a 84,30 pasos, la mejora de 422,40 a 416,70 puntos y la reducción de 84,28 % a 83,14 %. Afecta al README, a la sección 12 del notebook y a la sección 13 del notebook.
- `results/evaluation_summary.csv` y `results/baseline_vs_dqn.csv`, que no coincidían con `dqn_evaluation.csv`.
- Hitos de la curva de aprendizaje en el README, que correspondían a otra corrida.
- Conclusión del README, que mencionaba una semilla con menor estabilidad cuando las tres resolvieron todos los episodios.
- Limitación de la sección 12 del notebook, que decía que solo se había usado una semilla.
- Referencia al libro de Lapan, actualizada a la 3.ª edición (2024).

### Actualizado

- Referencias: se agregaron los artículos de DQN (2013), Double DQN, Dueling DQN y el libro de Sutton y Barto.
- Limitaciones metodológicas y reflexión sobre dificultades, con lo aprendido de la variación entre semillas.
- Estructura del repositorio y tabla de archivos de resultados en el README.

### Sin cambios

- Código y salidas de `Taller2_DQN_Acrobot.ipynb`.
- `seed_comparison.py`, `run_config.json`, `requirements.txt` y `.gitignore`.
- Resultados de `results/seed_comparison/`, `dqn_evaluation.csv`, `dqn_learning_curve.csv` y las figuras originales.

## Versión 1.0 — 11 de octubre de 2026

Entrega original del Taller 2: DQN con Stable-Baselines3 sobre `Acrobot-v1`, baseline aleatorio, evaluación de 30 episodios y comparación entre las semillas 42, 25 y 123.
