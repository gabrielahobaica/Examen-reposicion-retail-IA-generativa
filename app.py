# =============================================================================
# Tablero del sistema de reposición retail (Hito 2 · Caso E)
# IA Generativa para la Gestión Industrial · UDD
#
# Muestra 4 cosas:
#   1. Última recomendación del sistema (lo que el jefe de abastecimiento aprobó o rechazó)
#   2. Modelos de pronóstico en el tiempo (Holt-Winters, LSTM y los dos simples)
#   3. Señales de sala: góndola vacía (ViT) y reseñas negativas (BERT) por semana
#   4. Historial de recomendaciones del LLM con su decisión humana
#   5. Negocio: inventario vs. nivel de servicio, ROI con supuestos ajustables y control humano
#
# Datos:
#   - data/*.csv: resultados de los modelos (backtest y notebooks de Colab), fijos.
#   - Historial: hoja "historial" de Google Sheets publicada como CSV. La dirección va
#     en los "secrets" de Streamlit (HISTORIAL_CSV_URL), nunca en el código.
#   - Negocio: data/negocio_frontera.csv, negocio_corridas.csv y negocio_roi.csv, que
#     produce el notebook H2_resultados_sistema.ipynb, parte C (simulación de 2017 y ROI).
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


def pct(v, d=1):
    return f"{v * 100:.{d}f} %".replace(".", ",")


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

# Resultados de negocio (notebook H2_resultados_sistema.ipynb, parte C)
@st.cache_data
def leer_negocio():
    f = pd.read_csv("data/negocio_frontera.csv")
    c = pd.read_csv("data/negocio_corridas.csv", parse_dates=["fecha"], keep_default_na=False)
    r = pd.read_csv("data/negocio_roi.csv", dtype=str).fillna("")
    roi = {x["indicador"]: a_numero(x["valor"]) for _, x in r.iterrows()}
    return f, c, roi, "notebook H2_resultados_sistema (simulación de 2017)"


try:
    frontera, corridas, ROI, origen_negocio = leer_negocio()
except Exception as e:
    frontera, corridas, ROI, origen_negocio = pd.DataFrame(), pd.DataFrame(), {}, f"no se pudo leer: {e}"

# ---------------------------------------------------------------------------- encabezado
st.title("🛒 Reposición de tiendas con IA")
st.caption("Caso E · Retail y logística · Pronóstico (Holt-Winters y LSTM) + visión (ViT) + texto (BERT) + LLM con aprobación humana en n8n")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Última recomendación", "📈 Modelos de pronóstico", "🏪 Señales de sala",
                                        "🗂️ Historial y decisiones", "💼 Negocio"])

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

# ---------------------------------------------------------------------------- 5. negocio
with tab5:
    st.caption(f"Fuente: {origen_negocio}. Para recalcular con otros supuestos: correr el notebook y reemplazar los CSV de data/.")
    if frontera.empty or not ROI:
        st.warning("No hay resultados de negocio para mostrar.")
        st.stop()
    sup = lambda k, d: ROI.get(f"supuesto: {k}") if ROI.get(f"supuesto: {k}") is not None else d

    # 5a. beneficio medido en la simulación
    st.subheader("¿Cuánto inventario hace falta para el mismo nivel de servicio?")
    st.caption("Simulación de reposición semanal de 2017 (revisión cada lunes, pedido que llega en el plazo de cada SKU, "
               "venta perdida si no hay stock), 3 tiendas × 5 SKU. Cada punto es un stock de seguridad distinto (0 a 3 veces el normal). "
               "A = promedio móvil de 7 días; B = mejor de los 3 modelos del Hito 1; C = mejor de los 4 modelos (con LSTM).")
    COL_POL = {"A · práctica simple": "#667085", "B · Hito 1": "#1f4e79", "C · Hito 2 (LSTM)": "#e07b00"}
    fr = frontera[frontera["fill_rate"] < 0.9997]
    ns = sup("nivel_servicio_comparacion", 0.995)
    base_graf = alt.Chart(fr).encode(
        x=alt.X("fill_rate:Q", title="Nivel de servicio (fill rate)", scale=alt.Scale(zero=False), axis=alt.Axis(format=".1%")),
        y=alt.Y("inventario_prom_clp:Q", title="Inventario promedio", scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr="'$' + replace(format(datum.value / 1e6, '.1f'), '.', ',') + ' M'")),
        color=alt.Color("politica:N", title=None, scale=alt.Scale(domain=list(COL_POL), range=list(COL_POL.values()))),
        tooltip=["politica:N", alt.Tooltip("factor_seguridad:Q", title="factor"), alt.Tooltip("fill_rate:Q", format=".2%"),
                 alt.Tooltip("inventario_prom_clp:Q", format=",.0f"), alt.Tooltip("venta_perdida_u:Q", title="venta perdida (u)")])
    regla_ns = alt.Chart(pd.DataFrame({"x": [ns]})).mark_rule(strokeDash=[4, 4], color="#667085").encode(x="x:Q")
    st.altair_chart(base_graf.mark_line(point=True) + regla_ns, width="stretch")
    inv_A, inv_C = ROI.get("inventario_A_clp"), ROI.get("inventario_C_clp")
    if inv_A and inv_C:
        m_ = lambda v: f"${v / 1e6:.2f} M".replace(".", ",")
        st.write(f"Al {pct(ns)} de servicio (línea punteada) la política C necesita **{m_(inv_C)}** de inventario promedio y la A "
                 f"**{m_(inv_A)}**: {pct(1 - inv_C / inv_A)} menos (≈ ${miles(inv_A - inv_C)} de inventario; "
                 f"≈ ${miles(ROI.get('beneficio_inventario_clp', 0))} al año de costo de mantenerlo). "
                 "**El beneficio principal del sistema no es inventario, son horas de planificación.**")

    # 5b. ROI con supuestos ajustables (parten de los supuestos del notebook)
    st.subheader("ROI simple (piloto: 3 tiendas × 5 SKU, 1 año)")
    st.caption("Los controles parten de los supuestos del notebook; muévalos para ver la sensibilidad. "
               "El supuesto que más pesa son las horas que hoy toma armar los pedidos a mano.")
    a, b, c_, d_ = st.columns(4)
    h_manual = a.slider("Horas/semana armando pedidos hoy", 0.5, 8.0, float(sup("horas_manual_semana", 3.0)), 0.5)
    h_sist = b.slider("Horas/semana con el sistema", 0.0, 2.0, float(sup("horas_sistema_semana", 0.5)), 0.25)
    costo_h = c_.slider("Costo hora del jefe ($)", 5_000, 20_000, int(sup("costo_hora_clp", 10_000)), 500)
    plan = d_.radio("n8n", ["Cloud Starter", "Autoalojado ($0)"])
    AHORRO_INV = ROI.get("beneficio_inventario_clp") or 0.0
    N8N = ROI.get("costo_n8n_clp", 0.0) if plan.startswith("Cloud") else 0.0
    LLM = ROI.get("costo_llm_clp", 0.0)
    H_MANT, H_DESA = sup("horas_mantencion_anio", 45), sup("horas_desarrollo", 80)
    beneficio = max(h_manual - h_sist, 0) * 52 * costo_h + AHORRO_INV
    costo = N8N + LLM + H_MANT * costo_h
    inversion = H_DESA * costo_h
    neto = beneficio - costo
    m = st.columns(6)
    m[0].metric("Beneficio anual", "$" + miles(beneficio))
    m[1].metric("Costo operación anual", "$" + miles(costo))
    m[2].metric("Beneficio neto anual", "$" + miles(neto))
    m[3].metric("ROI año 1", pct((neto - inversion) / (costo + inversion), 0))
    m[4].metric("ROI en régimen", pct(neto / costo, 0) if costo > 0 else "-")
    m[5].metric("Payback", f"{inversion / neto:.1f} años".replace(".", ",") if neto > 0 else "no se recupera")
    with st.expander("Fórmulas y supuestos"):
        st.markdown(
            f"- **Beneficio** = (horas hoy − horas con sistema) × 52 × costo hora + ahorro de inventario (${miles(AHORRO_INV)}, de la simulación)\n"
            f"- **Costo** = n8n (${miles(N8N)}) + LLM (${miles(LLM)}: {miles(sup('corridas_anio', 104))} corridas al año) "
            f"+ mantención {miles(H_MANT)} h × costo hora\n"
            f"- **Inversión** = {miles(H_DESA)} h de desarrollo × costo hora = ${miles(inversion)}\n"
            "- **ROI año 1** = (neto − inversión) / (costo + inversión) · **ROI en régimen** = neto / costo · **Payback** = inversión / neto\n"
            "- Resultado del notebook con sus supuestos: "
            f"neto ${miles(ROI.get('beneficio_neto_clp', 0))}, ROI año 1 {pct(ROI.get('roi_anio1', 0), 0)}, ROI en régimen {pct(ROI.get('roi_regimen', 0), 0)}.\n"
            "- No incluye venta recuperada por detectar góndolas vacías (no se puede medir con estos datos).")

    # 5c. control humano
    st.subheader("Punto de control humano")
    izq, der = st.columns(2)
    with izq:
        n_l = int(ROI.get("lunes_simulados") or len(corridas))
        n_ap = int(ROI.get("lunes_requieren_aprobacion") or (corridas["requiere_aprobacion"] == "SI").sum())
        n_rel = ROI.get("lunes_aprobacion_umbral_relativo")
        st.markdown(f"**Simulación: los {n_l} lunes de 2017**")
        k1, k2 = st.columns(2)
        k1.metric("Pedirían aprobación (reglas actuales)", f"{n_ap} de {n_l} ({pct(n_ap / n_l, 0)})")
        if n_rel is not None:
            k2.metric("Con umbral relativo al pedido habitual", f"{int(n_rel)} de {n_l} ({pct(n_rel / n_l, 0)})",
                      help=f"Aprobar solo si el pedido supera {sup('factor_umbral_relativo', 1.3)} × la mediana de las 8 semanas previas "
                           "(se mantienen las reglas de error alto y venta anómala).")
        mot = corridas["motivos"].str.split("|").explode().str.strip()
        mot = mot[mot != ""].value_counts().reset_index()
        mot.columns = ["Motivo", "Lunes"]
        st.altair_chart(alt.Chart(mot).mark_bar(color="#1f4e79").encode(
            x="Lunes:Q", y=alt.Y("Motivo:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250))), width="stretch")
        st.caption("Con un umbral fijo en pesos casi todos los lunes piden aprobación: riesgo de fatiga de alertas "
                   "(el jefe aprueba sin leer). Un umbral relativo deja la aprobación para las semanas fuera de lo normal.")
    with der:
        st.markdown("**Corridas reales registradas**")
        if historial.empty:
            st.warning(f"Historial no disponible: {origen_hist}")
        else:
            rev = historial.get("requiere_revision_humana", pd.Series(dtype=str)) == "SI"
            st.metric("Corridas que pidieron aprobación", f"{int(rev.sum())} de {len(historial)} ({pct(rev.mean(), 0)})")
            if "respondido_en" in historial and "ejecutado_en" in historial:   # en vivo desde el historial
                ini = pd.to_datetime(historial["ejecutado_en"], errors="coerce").dt.tz_localize(
                    "America/Santiago", ambiguous="NaT", nonexistent="NaT")
                fin = pd.to_datetime(historial["respondido_en"], errors="coerce", utc=True)
                minutos = ((fin - ini).dt.total_seconds() / 60).dropna()
                minutos = minutos[(minutos >= 0) & (minutos <= 7 * 1440)]
                if len(minutos):
                    st.metric("Tiempo de respuesta del jefe (mediana)", f"{minutos.median():.0f} min",
                              help=f"{len(minutos)} respuestas medidas · límite de espera 4 h")
            est = historial["estado_aprobacion"].value_counts().reset_index()
            est.columns = ["Decisión", "Corridas"]
            st.dataframe(est, width="stretch", hide_index=True)

st.divider()
st.caption("Las cantidades las calcula el modelo; el LLM solo redacta y una validación en Python corrige lo que invente. "
           "Ningún pedido sobre los límites se emite sin aprobación del jefe de abastecimiento en Telegram.")
