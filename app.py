# =============================================================================
# Tablero del sistema de reposición retail (Hito 2 · Caso E)
# IA Generativa para la Gestión Industrial · UDD
#
# Muestra 4 cosas:
#   1. Última recomendación del sistema (lo que el jefe de abastecimiento aprobó o rechazó)
#   2. Modelos de pronóstico en el tiempo (Holt-Winters, LSTM y los dos simples)
#   3. Señales de sala: góndola vacía (ViT) y reseñas negativas (BERT) por semana
#   4. Historial de recomendaciones del LLM con su decisión humana
#
# Datos:
#   - data/*.csv: resultados de los modelos (backtest y notebooks de Colab), fijos.
#   - Historial: hoja "historial" de Google Sheets publicada como CSV. La dirección va
#     en los "secrets" de Streamlit (HISTORIAL_CSV_URL), nunca en el código.
# =============================================================================
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Reposición retail con IA", page_icon="🛒", layout="wide")

MODELOS = {"hw": "Holt-Winters", "lstm": "LSTM", "pm7": "Promedio móvil 7 d", "ingenuo": "Ingenuo estacional"}
COLORES = {"Holt-Winters": "#1f4e79", "LSTM": "#e07b00", "Promedio móvil 7 d": "#9aa5b1", "Ingenuo estacional": "#c9ced6"}


def a_numero(v):
    """Convierte '0,1205', '15.575.040' o 0.12 en número (Sheets en Chile usa coma decimal)."""
    if pd.isna(v) or v == "":
        return None
    s = str(v).strip().replace("$", "").replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def miles(v):
    return f"{int(round(v)):,}".replace(",", ".") if v is not None else "-"


# ---------------------------------------------------------------------------- datos
@st.cache_data
def leer_fijos():
    bt = pd.read_csv("data/backtest_pronostico.csv", parse_dates=["corte"])
    img = pd.read_csv("data/modelo_imagen_semanal.csv", parse_dates=["fecha"])
    txt = pd.read_csv("data/modelo_texto_semanal.csv", parse_dates=["fecha"])
    comp_img = pd.read_csv("data/comparacion_imagenes.csv", index_col=0)
    comp_txt = pd.read_csv("data/comparacion_texto.csv", index_col=0)
    return bt, img, txt, comp_img, comp_txt


@st.cache_data(ttl=300)   # se vuelve a leer cada 5 minutos
def leer_historial(url):
    h = pd.read_csv(url, dtype=str)
    for c in ["unidades_a_pedir", "valor_pedido_clp", "skus_riesgo_alto", "wape_hw", "wape_lstm", "wape_ingenuo", "wape_pm7"]:
        if c in h:
            h[c] = h[c].map(a_numero)
    if "estado_aprobacion" not in h:
        h["estado_aprobacion"] = "sin registro (Hito 1)"
    h["estado_aprobacion"] = h["estado_aprobacion"].fillna("sin registro (Hito 1)")
    return h


bt, img, txt, comp_img, comp_txt = leer_fijos()
try:
    url = st.secrets.get("HISTORIAL_CSV_URL", "")
except Exception:          # sin secrets configurados (por ejemplo, al correrlo en el computador)
    url = ""
try:
    historial = leer_historial(url) if url else leer_historial("data/historial_ejemplo.csv")
    origen_hist = "Google Sheets (en vivo)" if url else "archivo de ejemplo (falta configurar HISTORIAL_CSV_URL)"
except Exception as e:
    historial, origen_hist = pd.DataFrame(), f"no se pudo leer el historial: {e}"

# ---------------------------------------------------------------------------- encabezado
st.title("🛒 Reposición de tiendas con IA")
st.caption("Caso E · Retail y logística · Pronóstico (Holt-Winters y LSTM) + visión (ViT) + texto (BERT) + LLM con aprobación humana en n8n")

tab1, tab2, tab3, tab4 = st.tabs(["📌 Última recomendación", "📈 Modelos de pronóstico", "🏪 Señales de sala", "🗂️ Historial y decisiones"])

# ---------------------------------------------------------------------------- 1. última recomendación
with tab1:
    if historial.empty:
        st.warning(f"Historial no disponible: {origen_hist}")
    else:
        u = historial.iloc[-1]
        st.caption(f"Fuente: {origen_hist} · ejecutado {u.get('ejecutado_en', '')}")
        c = st.columns(5)
        c[0].metric("Fecha de corte", u.get("fecha_corte", "-"))
        c[1].metric("Prioridad", str(u.get("prioridad", "-")).upper())
        c[2].metric("Unidades a pedir", miles(u.get("unidades_a_pedir")))
        c[3].metric("Valor del pedido", "$" + miles(u.get("valor_pedido_clp")))
        c[4].metric("Combinaciones en riesgo alto", miles(u.get("skus_riesgo_alto")))
        st.info(f"**Decisión humana:** {u.get('estado_aprobacion', '-')}"
                + (f" · motivos: {u['motivos_revision']}" if isinstance(u.get("motivos_revision"), str) and u.get("motivos_revision") else ""))
        st.markdown(f"**Diagnóstico.** {u.get('diagnostico', '')}")
        st.markdown(f"**Recomendación.** {u.get('recomendacion', '')}")
        st.markdown(f"**Justificación.** {u.get('justificacion', '')}")
        ETIQUETAS = {"modelos_elegidos": "Modelos elegidos", "gondola_vacia": "Góndola vacía",
                     "resenas_negativas": "Reseñas negativas", "discrepancias_gondola": "Discrepancias de sala"}
        extra = {ETIQUETAS[k]: u.get(k) for k in ETIQUETAS if isinstance(u.get(k), str)}
        if extra:
            st.markdown(" · ".join(f"**{k}:** {v}" for k, v in extra.items()))

# ---------------------------------------------------------------------------- 2. modelos de pronóstico
with tab2:
    st.subheader("Error de pronóstico (WAPE) en el backtest semanal de 2017")
    st.caption("Cada punto es el promedio de las 15 combinaciones tienda-SKU en ese corte: 4 semanas de prueba, pronóstico a 7 días. "
               "La LSTM se entrenó solo con 2016; los cortes de enero (antes del 28-01) se excluyen porque su prueba toca 2016.")
    justo = bt[bt["corte"] >= "2017-01-28"]
    col = st.columns(4)
    for i, (k, nombre) in enumerate(MODELOS.items()):
        col[i].metric(nombre, f"{justo[k].mean() * 100:.1f} %".replace(".", ","))
    st.write(f"La LSTM tiene menor error que Holt-Winters en **{(justo['lstm'] < justo['hw']).mean():.0%}** de los "
             f"{len(justo)} casos (corte × tienda × SKU).")

    tiendas = ["Todas"] + sorted(bt["tienda"].unique())
    t_sel = st.selectbox("Tienda", tiendas)
    d = justo if t_sel == "Todas" else justo[justo["tienda"] == t_sel]
    largo = d.groupby("corte")[list(MODELOS)].mean().reset_index().melt("corte", var_name="m", value_name="wape")
    largo["Modelo"] = largo["m"].map(MODELOS)
    graf = alt.Chart(largo).mark_line(point=True).encode(
        x=alt.X("corte:T", title="Fecha de corte", axis=alt.Axis(format="%m-%Y")),
        y=alt.Y("wape:Q", title="WAPE (menor es mejor)", axis=alt.Axis(format="%")),
        color=alt.Color("Modelo:N", scale=alt.Scale(domain=list(COLORES), range=list(COLORES.values()))),
        tooltip=["corte:T", "Modelo:N", alt.Tooltip("wape:Q", format=".1%")])
    st.altair_chart(graf, width="stretch")

    st.subheader("Error por SKU")
    por_sku = d.groupby("sku")[list(MODELOS)].mean().rename(columns=MODELOS)
    st.dataframe(por_sku.style.format("{:.1%}").highlight_min(axis=1, color="#d1fadf"), width="stretch")

# ---------------------------------------------------------------------------- 3. señales de sala
with tab3:
    a, b = st.columns(2)
    with a:
        st.subheader("Imágenes: góndola vacía")
        st.caption("Modelos comparados con recortes de fotos que nunca vieron (5.653 recortes).")
        ti = comp_img[["accuracy_balanceada", "precision_vacio", "recall_vacio", "f1_vacio", "ms_por_recorte_cpu"]].copy()
        ti.columns = ["Accuracy balanceada", "Precisión vacío", "Recall vacío", "F1 vacío", "ms/recorte (CPU)"]
        ti["ms/recorte (CPU)"] = ti["ms/recorte (CPU)"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        st.dataframe(ti.style.format("{:.3f}", subset=ti.columns[:4]), width="stretch")
    with b:
        st.subheader("Texto: reseñas de clientes")
        st.caption("3.000 reseñas balanceadas (1.000 por clase); la etiqueta real son las estrellas del cliente.")
        tt = comp_txt.copy()
        tt.columns = ["Accuracy", "F1 macro", "ms/reseña (CPU)"]
        tt["ms/reseña (CPU)"] = tt["ms/reseña (CPU)"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        st.dataframe(tt.style.format("{:.3f}", subset=["Accuracy", "F1 macro"]), width="stretch")

    t_sel3 = st.selectbox("Tienda ", sorted(img["tienda"].unique()))
    gi = img[img["tienda"] == t_sel3].melt("fecha", ["pct_vacia_vit", "pct_vacia_regla", "pct_hueco_real_mascara"], "serie", "pct")
    gi["serie"] = gi["serie"].map({"pct_vacia_vit": "ViT (modelo elegido)", "pct_vacia_regla": "Regla textura+color", "pct_hueco_real_mascara": "Real (máscara)"})
    st.markdown("**% de la góndola vacía por semana** (sobre 15 % se considera quiebre visible)")
    linea_umbral = alt.Chart(pd.DataFrame({"y": [0.15]})).mark_rule(strokeDash=[4, 4], color="#B42318").encode(y="y:Q")
    st.altair_chart(alt.Chart(gi).mark_line().encode(
        x=alt.X("fecha:T", title="Lunes", axis=alt.Axis(format="%m-%Y")), y=alt.Y("pct:Q", title="% vacía", axis=alt.Axis(format="%")),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=["Real (máscara)", "ViT (modelo elegido)", "Regla textura+color"],
                                                               range=["#111827", "#e07b00", "#b8c0cc"])),
        tooltip=["fecha:T", "serie:N", alt.Tooltip("pct:Q", format=".0%")]) + linea_umbral,
        width="stretch")

    gt = txt[txt["tienda"] == t_sel3].melt("fecha", ["pct_negativas", "pct_negativas_regla", "pct_negativas_real"], "serie", "pct")
    gt["serie"] = gt["serie"].map({"pct_negativas": "BERT (modelo elegido)", "pct_negativas_regla": "Regla de palabras", "pct_negativas_real": "Real (estrellas)"})
    st.markdown("**% de reseñas negativas por semana** (6 reseñas por tienda y semana; asignación a tiendas simulada)")
    st.altair_chart(alt.Chart(gt).mark_line().encode(
        x=alt.X("fecha:T", title="Lunes", axis=alt.Axis(format="%m-%Y")), y=alt.Y("pct:Q", title="% negativas", axis=alt.Axis(format="%")),
        color=alt.Color("serie:N", title=None, scale=alt.Scale(domain=["Real (estrellas)", "BERT (modelo elegido)", "Regla de palabras"],
                                                               range=["#111827", "#1f4e79", "#b8c0cc"])),
        tooltip=["fecha:T", "serie:N", alt.Tooltip("pct:Q", format=".0%")]),
        width="stretch")

# ---------------------------------------------------------------------------- 4. historial
with tab4:
    if historial.empty:
        st.warning(f"Historial no disponible: {origen_hist}")
    else:
        st.caption(f"Fuente: {origen_hist} · {len(historial)} corridas registradas")
        c = st.columns(3)
        c[0].metric("Corridas", len(historial))
        c[1].metric("Requirieron revisión humana", int((historial.get("requiere_revision_humana", pd.Series(dtype=str)) == "SI").sum()))
        c[2].metric("Usaron respaldo por reglas (LLM caído)", int((historial.get("fuente_recomendacion", pd.Series(dtype=str)) == "reglas_respaldo").sum()))
        estados = historial["estado_aprobacion"].value_counts().reset_index()
        estados.columns = ["Decisión", "Corridas"]
        st.altair_chart(alt.Chart(estados).mark_bar().encode(x="Corridas:Q", y=alt.Y("Decisión:N", sort="-x", title=None)),
                        width="stretch")
        cols = [c for c in ["ejecutado_en", "origen", "fecha_corte", "tienda_filtro", "prioridad", "unidades_a_pedir", "valor_pedido_clp",
                            "estado_aprobacion", "modelos_elegidos", "gondola_vacia", "recomendacion"] if c in historial]
        st.dataframe(historial[cols].iloc[::-1], width="stretch", hide_index=True)

st.divider()
st.caption("Las cantidades las calcula el modelo; el LLM solo redacta y una validación en Python corrige lo que invente. "
           "Ningún pedido sobre los límites se emite sin aprobación del jefe de abastecimiento en Telegram.")
