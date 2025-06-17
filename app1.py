# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 17:52:52 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Candidato - Subgerente de Regulación CENAGAS", layout="wide")

# Título principal
st.title("📊 Candidato a Subgerente de Regulación en CENAGAS")
st.markdown("""
### Javier Horacio Pérez Ricárdez  
**Matemático | Maestro en Ciencias de la Computación | Doctorando en IA**  
Especialista en análisis regulatorio, ciencia de datos e inteligencia artificial aplicada al sector energético.  
📍 Ciudad de México | ✉️ jahoperi@gmail.com | 📞 +52 56 1056 4095
""")

# Perfil profesional
st.header("🧠 Perfil Profesional")
st.markdown("""
Soy Matemático con Maestría en Ciencias de la Computación y actualmente curso el **segundo semestre del Doctorado en Inteligencia Artificial** en la Universidad Panamericana, Campus Mixcoac (de 7 pm a 10 pm).

Mi formación me permite integrar una visión **analítica, computacional y estratégica** en la gestión regulatoria y técnica del sector energético, particularmente en gas natural.

Estoy comprometido con el servicio público y con el fortalecimiento institucional del mercado energético a través de tecnologías emergentes, automatización normativa y visualización de datos.
""")

# Análisis técnico-regulatorio
st.header("📈 Análisis Técnico-Regulatorio del Gas Natural")
st.markdown("""
La gestión eficiente del SISTRANGAS requiere monitoreo continuo y análisis de datos.  
Aquí presento una visualización simulada de la demanda frente a la capacidad operativa:
""")

data = pd.DataFrame({
    "Mes": pd.date_range("2024-01-01", periods=12, freq='M'),
    "Demanda estimada (GJ)": [4200, 4500, 4700, 4900, 4600, 4400, 4300, 4100, 4200, 4600, 4800, 5000],
    "Capacidad disponible (GJ)": [5200, 5000, 5100, 5150, 5100, 5050, 4950, 4900, 5000, 5200, 5300, 5400]
})

fig = px.line(data, x="Mes", y=["Demanda estimada (GJ)", "Capacidad disponible (GJ)"],
              title="Demanda vs Capacidad del SISTRANGAS (Simulación)", markers=True)
st.plotly_chart(fig, use_container_width=True)

# Sección IA: Monitoreo normativo
st.header("🤖 Algoritmos de IA aplicados a Monitoreo Normativo")
st.markdown("""
La siguiente simulación muestra cómo un modelo de IA puede **predecir el riesgo de incumplimiento normativo** a partir de variables como el retraso en reportes, frecuencia de auditorías, y cumplimiento histórico.
""")

# Generar datos simulados
np.random.seed(42)
df_ai = pd.DataFrame({
    "Días de retraso": np.random.randint(0, 30, 100),
    "Auditorías previas": np.random.randint(0, 10, 100),
    "Porcentaje de cumplimiento histórico": np.random.uniform(60, 100, 100),
})
df_ai["Incumplimiento"] = ((df_ai["Días de retraso"] > 10) & (df_ai["Porcentaje de cumplimiento histórico"] < 85)).astype(int)

# Entrenamiento del modelo
X = df_ai[["Días de retraso", "Auditorías previas", "Porcentaje de cumplimiento histórico"]]
y = df_ai["Incumplimiento"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = DecisionTreeClassifier(max_depth=4)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)
accuracy = accuracy_score(y_test, predicciones)

st.success(f"🔍 Precisión del modelo de predicción de incumplimiento: {accuracy:.2%}")

# Interfaz de simulación
st.markdown("### 🔬 Simula una situación de cumplimiento:")
col1, col2, col3 = st.columns(3)
dias_retraso = col1.slider("Días de retraso", 0, 30, 5)
auditorias_previas = col2.slider("Auditorías previas", 0, 10, 2)
cumplimiento_hist = col3.slider("Cumplimiento histórico (%)", 60, 100, 90)

input_usuario = pd.DataFrame({
    "Días de retraso": [dias_retraso],
    "Auditorías previas": [auditorias_previas],
    "Porcentaje de cumplimiento histórico": [cumplimiento_hist]
})
riesgo = modelo.predict(input_usuario)[0]

if riesgo == 1:
    st.error("⚠️ Riesgo alto de INCUMPLIMIENTO normativo detectado.")
else:
    st.success("✅ Cumplimiento normativo dentro de parámetros aceptables.")

# Mostrar datos simulados
st.markdown("### 📄 Datos utilizados para el modelo de IA de cumplimiento normativo")
st.dataframe(df_ai.head(20), use_container_width=True)

# Botón de descarga del DataFrame
csv = df_ai.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Descargar datos simulados en CSV",
    data=csv,
    file_name="datos_ia_cumplimiento.csv",
    mime="text/csv"
)

# Propuesta regulatoria
st.header("🔍 Propuesta de Mejora Regulatoria Basada en IA y Datos")
st.markdown("""
Propongo una solución de monitoreo normativo inteligente que incluya:

1. **Modelos de aprendizaje supervisado** para alertas de incumplimiento.
2. **Dashboards de trazabilidad normativa** actualizados en tiempo real.
3. **Análisis de riesgos regulatorios por IA** para tomar decisiones preventivas.
4. **Mapeo dinámico de obligaciones por área, riesgo y frecuencia**.

Esto permitirá un enfoque preventivo, basado en datos, transparente y automatizado.
""")

# Aplicaciones desarrolladas
with st.expander("🧩 Aplicaciones desarrolladas"):
    st.markdown("""
- 📊 Dashboards para visualización de KPIs regulatorios en energía.
- 🤖 Modelos de predicción de demanda y anomalías en sistemas técnicos.
- 🧠 Algoritmos de IA aplicados a monitoreo normativo.
- 🛠️ Automatización de procesos regulatorios con Python.
""")

# CV y contacto
st.header("📄 CV y Contacto")

with open("CV_Javier_Horacio_Perez_Ricardez.pdf", "rb") as file:
    st.download_button(
        label="📥 Descargar CV en PDF",
        data=file,
        file_name="CV_Javier_Perez_CENAGAS.pdf",
        mime="application/pdf"
    )

st.markdown("""
📧 Correo: [jahoperi@gmail.com](mailto:jahoperi@gmail.com)  
📞 Teléfono: +52 56 1056 4095  
📍 Ciudad de México  
🎓 Doctorado en IA: Universidad Panamericana – Campus Mixcoac, de 7 pm a 10 pm  
""")

st.markdown("---")
st.caption("Aplicación desarrollada por Javier Horacio Pérez Ricárdez para la vacante de Subgerente de Regulación en CENAGAS – 2025.")

