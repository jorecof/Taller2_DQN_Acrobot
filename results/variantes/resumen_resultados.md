# Resumen de resultados — Acrobot-v1

Semillas de entrenamiento: [42, 25, 123] · pasos: 100,000 · episodios de evaluación por semilla: 30

Política aleatoria: retorno medio -496.67, longitud 496.7, éxito 3%

| Agente | Retorno medio | IC 95 % | Std entre semillas | Peor episodio | Longitud media | Éxito | Mejora vs aleatorio |
|---|---|---|---|---|---|---|---|
| Dueling DQN | -73.08 | [-75.71, -70.45] | 1.06 | -103 | 74.1 | 100% | 423.59 |
| SB3 DQN | -79.11 | [-91.86, -66.36] | 5.13 | -117 | 80.1 | 100% | 417.56 |
| DQN propio | -79.20 | [-79.42, -78.98] | 0.09 | -141 | 80.2 | 100% | 417.47 |
| Double DQN | -96.00 | [-154.12, -37.88] | 23.40 | -500 | 97.0 | 97% | 400.67 |

## Por semilla

| Agente | Semilla | Retorno medio ± std | Mín | Máx | Longitud | Éxito |
|---|---|---|---|---|---|---|
| DQN propio | 25 | -79.10 ± 11.49 | -113 | -62 | 80.1 | 100% |
| DQN propio | 42 | -79.27 ± 15.77 | -141 | -63 | 80.3 | 100% |
| DQN propio | 123 | -79.23 ± 10.47 | -107 | -62 | 80.2 | 100% |
| Double DQN | 25 | -81.73 ± 12.54 | -109 | -62 | 82.7 | 100% |
| Double DQN | 42 | -83.27 ± 13.45 | -114 | -62 | 84.3 | 100% |
| Double DQN | 123 | -123.00 ± 128.77 | -500 | -62 | 123.9 | 90% |
| Dueling DQN | 25 | -73.83 ± 10.79 | -103 | -62 | 74.8 | 100% |
| Dueling DQN | 42 | -71.87 ± 8.63 | -96 | -62 | 72.9 | 100% |
| Dueling DQN | 123 | -73.53 ± 7.64 | -94 | -62 | 74.5 | 100% |
| SB3 DQN | 25 | -77.70 ± 13.99 | -117 | -62 | 78.7 | 100% |
| SB3 DQN | 42 | -74.83 ± 9.26 | -106 | -62 | 75.8 | 100% |
| SB3 DQN | 123 | -84.80 ± 12.38 | -109 | -62 | 85.8 | 100% |

## Lectura automática

- Mejor agente: **Dueling DQN** (-73.08); peor: Double DQN (-96.00).
- Los IC 95 % del mejor y el peor se solapan: con estas semillas la diferencia NO es concluyente.
- Todos los agentes superan el umbral de −100: sí.
