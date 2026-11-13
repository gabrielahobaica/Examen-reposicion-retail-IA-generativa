# Tablero · Reposición de tiendas con IA (Hito 2, Caso E)

Tablero en Streamlit del sistema de reposición retail del curso *IA Generativa para la Gestión Industrial* (UDD).

| Pestaña | Qué muestra | De dónde salen los datos |
|---|---|---|
| Última recomendación | Fecha de corte, prioridad, pedido, decisión del jefe de abastecimiento y texto del LLM | Hoja `historial` de Google Sheets (en vivo) |
| Modelos de pronóstico | WAPE por semana de Holt-Winters, LSTM y los dos modelos simples; error por SKU | `data/backtest_pronostico.csv` |
| Señales de sala | % de góndola vacía (ViT) y % de reseñas negativas (BERT) por tienda y semana, contra el valor real | `data/modelo_imagen_semanal.csv`, `data/modelo_texto_semanal.csv` |
| Historial y decisiones | Todas las corridas de n8n, con su aprobación o rechazo | Hoja `historial` de Google Sheets (en vivo) |

## Archivos

```
app.py                         la aplicación
requirements.txt               librerías (streamlit, pandas, altair)
data/                          resultados de los modelos (fijos)
data/historial_ejemplo.csv     se usa solo si no se configuró la hoja en vivo
.streamlit/secrets.toml.example  modelo de la configuración secreta (no tiene datos reales)
```

## Probarlo en el computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos y supuestos

- **Pronóstico:** backtest de 2017, con 4 semanas de prueba y un horizonte de 7 días. La LSTM se entrenó solo con 2016, así que en el tablero se excluyen los cortes anteriores al 28-01-2017.
- **Imágenes:** fotos de góndolas del repositorio *gapDetection* (Santra, Ghosh y Mukherjee, 2022). Su asignación a las tiendas T01, T02 y T05 y a cada semana es **simulada**.
- **Reseñas:** *Amazon Fine Food Reviews* (SNAP, Stanford; McAuley y Leskovec, 2013). Su asignación a tiendas y semanas también es **simulada**.
- El repositorio no contiene credenciales ni datos personales.
