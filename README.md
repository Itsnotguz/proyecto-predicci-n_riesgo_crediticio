# Predicción de Riesgo Crediticio Bancario

Proyecto final del Bootcamp de Ciencia de Datos. Modelo de Machine Learning para predecir si un cliente bancario incumplirá (default) el pago de un préstamo, a partir de sus características socioeconómicas y crediticias.

## Contenido

- `credit_risk_pipeline.py` — pipeline completo: generación de datos, EDA, preprocesamiento, entrenamiento y evaluación de modelos.
- `data/clientes_credito.csv` — dataset de clientes (se genera automáticamente al correr el script).
- `plots/` — gráficos de EDA y evaluación (se generan automáticamente).
- `outputs/resumen_resultados.json` — resumen de métricas (se genera automáticamente).
- `Prediccion_Riesgo_Crediticio.pptx` — presentación del proyecto.

## Metodología

1. **Datos**: dataset simulado de 4,000 clientes con variables demográficas, financieras y crediticias.
2. **EDA**: análisis de distribución del target, relación entre historial crediticio y default, correlaciones.
3. **Preprocesamiento**: codificación de variables categóricas, escalado, split train/test estratificado (75/25).
4. **Modelos**: Regresión Logística y Random Forest (300 árboles), ambos con balanceo de clases.
5. **Evaluación**: accuracy, precisión, recall, F1-score, ROC-AUC, matrices de confusión e importancia de variables.

## Resultados principales

| Métrica    | Regresión Logística | Random Forest |
|------------|:--------------------:|:--------------:|
| Accuracy   | 0.65                 | 0.68           |
| Precisión  | 0.55                 | 0.59           |
| Recall     | 0.65                 | 0.61           |
| F1-score   | 0.59                 | 0.60           |
| ROC-AUC    | 0.691                | 0.722          |

**Random Forest** obtuvo el mejor desempeño general. Las variables más importantes fueron el **historial crediticio**, los **ingresos mensuales** y el **monto del préstamo**.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
proyecto_crediticio.py
```

Esto genera el dataset, los gráficos y el resumen de resultados en las carpetas `data/`, `plots/` y `outputs/`.

## Autor

Juan
