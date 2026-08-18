"""
================================================================================
PROYECTO FINAL - BOOTCAMP DE CIENCIA DE DATOS
Predicción de Riesgo Crediticio (Loan Default Prediction)
================================================================================
Autor: Juan
Descripción:
    Este proyecto construye un modelo de Machine Learning capaz de predecir
    si un cliente bancario incurrirá en incumplimiento (default) de un
    préstamo, a partir de sus características socioeconómicas y crediticias.

    Flujo del proyecto:
      1. Generación / carga del dataset
      2. Análisis Exploratorio de Datos (EDA)
      3. Preprocesamiento (limpieza, codificación, escalado)
      4. Entrenamiento de modelos (Regresión Logística y Random Forest)
      5. Evaluación y comparación de modelos
      6. Importancia de variables y conclusiones de negocio
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

sns.set_style("whitegrid")
PALETTE = ["#1B4965", "#5FA8D3", "#BEE9E8", "#E63946", "#2A9D8F"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

# Crea las carpetas si no existen (evita el error "non-existent directory")
for _dir in (DATA_DIR, PLOTS_DIR, OUT_DIR):
    os.makedirs(_dir, exist_ok=True)

# ==============================================================================
# 1. GENERACIÓN DEL DATASET
# ==============================================================================
# Se simula una base de clientes bancarios con variables típicas usadas en
# modelos reales de scoring crediticio (edad, ingresos, historial de pagos,
# monto del préstamo, deuda actual, tipo de empleo, etc.). La variable objetivo
# 'default' se genera con una lógica de riesgo realista para que el problema
# sea aprendible por los modelos (no aleatorio puro).

def generar_dataset(n=4000, seed=42):
    rng = np.random.default_rng(seed)

    edad = rng.integers(21, 70, n)
    ingresos_mensuales = rng.gamma(shape=6, scale=650, size=n).round(0) + 400
    antiguedad_laboral = rng.integers(0, 30, n)  # años
    monto_prestamo = rng.gamma(shape=3, scale=2500, size=n).round(0) + 500
    plazo_meses = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.15, 0.25, 0.3, 0.2, 0.1])
    historial_crediticio = rng.choice(
        ["Bueno", "Regular", "Malo"], size=n, p=[0.55, 0.30, 0.15]
    )
    tipo_empleo = rng.choice(
        ["Formal", "Independiente", "Informal"], size=n, p=[0.5, 0.3, 0.2]
    )
    num_creditos_activos = rng.integers(0, 6, n)
    deuda_actual = (rng.random(n) * 0.6 * ingresos_mensuales * 3).round(0)
    tiene_vivienda_propia = rng.choice([1, 0], size=n, p=[0.4, 0.6])

    # Ratio cuota/ingreso (proxy de capacidad de pago)
    cuota_mensual = monto_prestamo / plazo_meses
    ratio_cuota_ingreso = cuota_mensual / ingresos_mensuales

    hist_map = {"Bueno": 0.0, "Regular": 0.5, "Malo": 1.0}
    empleo_map = {"Formal": 0.0, "Independiente": 0.4, "Informal": 0.8}

    # --- Lógica de riesgo (score latente) ---
    score_riesgo = (
        2.4 * ratio_cuota_ingreso
        + 1.8 * pd.Series(historial_crediticio).map(hist_map).values
        + 1.1 * pd.Series(tipo_empleo).map(empleo_map).values
        + 0.35 * (deuda_actual / (ingresos_mensuales * 3 + 1))
        + 0.25 * (num_creditos_activos / 6)
        - 0.9 * tiene_vivienda_propia
        - 0.02 * antiguedad_laboral
        - 0.01 * (edad - 21) / 10
        + rng.normal(0, 0.55, n)  # ruido
    )

    prob_default = 1 / (1 + np.exp(-(score_riesgo - 1.1)))
    default = (rng.random(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "edad": edad,
        "ingresos_mensuales": ingresos_mensuales,
        "antiguedad_laboral": antiguedad_laboral,
        "tipo_empleo": tipo_empleo,
        "monto_prestamo": monto_prestamo,
        "plazo_meses": plazo_meses,
        "historial_crediticio": historial_crediticio,
        "num_creditos_activos": num_creditos_activos,
        "deuda_actual": deuda_actual,
        "tiene_vivienda_propia": tiene_vivienda_propia,
        "ratio_cuota_ingreso": ratio_cuota_ingreso.round(3),
        "default": default,
    })
    return df


print("=" * 80)
print("1. GENERANDO DATASET SINTÉTICO DE CLIENTES BANCARIOS")
print("=" * 80)
df = generar_dataset()
df.to_csv(os.path.join(DATA_DIR, "clientes_credito.csv"), index=False)
print(f"Dataset generado: {df.shape[0]} clientes, {df.shape[1]} columnas")
print(f"Tasa de default (clase positiva): {df['default'].mean():.2%}")
print(df.head(), "\n")

# ==============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
print("=" * 80)
print("2. ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 80)
print(df.describe(include="all").T, "\n")
print("Valores nulos por columna:\n", df.isnull().sum(), "\n")

# --- Gráfico 1: Distribución de la variable objetivo ---
plt.figure(figsize=(6, 5))
counts = df["default"].value_counts().sort_index()
plt.bar(["No Default (0)", "Default (1)"], counts.values, color=[PALETTE[0], PALETTE[3]])
for i, v in enumerate(counts.values):
    plt.text(i, v + 80, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=11, fontweight="bold")
plt.title("Distribución de la variable objetivo (Default)", fontsize=13, fontweight="bold")
plt.ylabel("Número de clientes")
plt.ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_distribucion_target.png"), dpi=150)
plt.close()

# --- Gráfico 2: Default por historial crediticio ---
plt.figure(figsize=(7, 5))
tasa_hist = df.groupby("historial_crediticio")["default"].mean().reindex(["Bueno", "Regular", "Malo"])
plt.bar(tasa_hist.index, tasa_hist.values, color=PALETTE[1])
for i, v in enumerate(tasa_hist.values):
    plt.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
plt.title("Tasa de default según historial crediticio", fontsize=13, fontweight="bold")
plt.ylabel("Tasa de default")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_default_por_historial.png"), dpi=150)
plt.close()

# --- Gráfico 3: Ratio cuota/ingreso vs default ---
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="default", y="ratio_cuota_ingreso",
            hue="default", palette=[PALETTE[0], PALETTE[3]], legend=False)
plt.xticks([0, 1], ["No Default", "Default"])
plt.title("Ratio cuota/ingreso según estado de default", fontsize=13, fontweight="bold")
plt.xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_ratio_cuota_ingreso.png"), dpi=150)
plt.close()

# --- Gráfico 4: Matriz de correlación (numéricas) ---
plt.figure(figsize=(8, 6))
num_cols = ["edad", "ingresos_mensuales", "antiguedad_laboral", "monto_prestamo",
            "plazo_meses", "num_creditos_activos", "deuda_actual",
            "ratio_cuota_ingreso", "default"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlación", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_matriz_correlacion.png"), dpi=150)
plt.close()

print("EDA completado. Gráficos guardados en /plots\n")

# ==============================================================================
# 3. PREPROCESAMIENTO
# ==============================================================================
print("=" * 80)
print("3. PREPROCESAMIENTO DE DATOS")
print("=" * 80)

df_model = df.copy()
le_dict = {}
for col in ["tipo_empleo", "historial_crediticio"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Codificación '{col}': {le_dict[col]}")

X = df_model.drop(columns=["default"])
y = df_model["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} registros | Test: {X_test.shape[0]} registros\n")

# ==============================================================================
# 4. ENTRENAMIENTO DE MODELOS
# ==============================================================================
print("=" * 80)
print("4. ENTRENAMIENTO DE MODELOS")
print("=" * 80)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
)
rf.fit(X_train, y_train)

print("Modelos entrenados: Regresión Logística y Random Forest\n")

# ==============================================================================
# 5. EVALUACIÓN DE MODELOS
# ==============================================================================
print("=" * 80)
print("5. EVALUACIÓN Y COMPARACIÓN DE MODELOS")
print("=" * 80)

resultados = {}

def evaluar_modelo(nombre, y_true, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    resultados[nombre] = metrics
    print(f"\n--- {nombre} ---")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))
    return metrics

y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
evaluar_modelo("Regresión Logística", y_test, y_pred_lr, y_proba_lr)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
evaluar_modelo("Random Forest", y_test, y_pred_rf, y_proba_rf)

# --- Gráfico 5: Curvas ROC comparadas ---
plt.figure(figsize=(7, 6))
for nombre, y_proba, color in [
    ("Regresión Logística", y_proba_lr, PALETTE[0]),
    ("Random Forest", y_proba_rf, PALETTE[3]),
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = resultados[nombre]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc:.3f})", color=color, linewidth=2.2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC — Comparación de modelos", fontsize=13, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_curva_roc.png"), dpi=150)
plt.close()

# --- Gráfico 6: Matrices de confusión ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (nombre, y_pred) in zip(axes, [("Regresión Logística", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
    ax.set_title(nombre, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
plt.suptitle("Matrices de confusión", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_matrices_confusion.png"), dpi=150)
plt.close()

# --- Gráfico 7: Importancia de variables (Random Forest) ---
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(importancias.index, importancias.values, color=PALETTE[1])
plt.title("Importancia de variables — Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_importancia_variables.png"), dpi=150)
plt.close()

print("\nGráficos de evaluación guardados en /plots\n")

# ==============================================================================
# 6. RESUMEN FINAL Y CONCLUSIONES
# ==============================================================================
print("=" * 80)
print("6. RESUMEN FINAL")
print("=" * 80)

mejor_modelo = max(resultados, key=lambda k: resultados[k]["roc_auc"])
print(f"Mejor modelo según ROC-AUC: {mejor_modelo} (AUC = {resultados[mejor_modelo]['roc_auc']:.4f})")

top_variables = importancias.sort_values(ascending=False).head(3)
print("Variables más importantes para predecir el riesgo:")
for var, val in top_variables.items():
    print(f"  - {var}: {val:.3f}")

resumen = {
    "n_registros": int(df.shape[0]),
    "tasa_default": float(df["default"].mean()),
    "resultados_modelos": resultados,
    "mejor_modelo": mejor_modelo,
    "top_3_variables": {k: float(v) for k, v in top_variables.items()},
}
with open(os.path.join(OUT_DIR, "resumen_resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\nResumen guardado en outputs/resumen_resultados.json")
print("=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)