import pandas as pd
import streamlit as st

from ai.disease_risk import assess_forecast
from ai.weather import get_forecast_with_meta
from core import auth
from core.helpers import live_data_caption, plot_options, risk_badge
from core import db

user = auth.require_login()
owner_id = user["id"]

st.title("🌦️ Weather & Disease Risk")
st.caption("7-day forecast with mulberry-specific disease/pest/spray guidance, using free Open-Meteo data.")

plots = plot_options(owner_id)
if plots:
    plot_id = st.selectbox("Plot", options=list(plots.keys()), format_func=lambda k: plots[k])
    plot = db.fetch_one("farm_plot", "id = ?", (plot_id,))
    lat, lon = plot["latitude"] or 19.07, plot["longitude"] or 74.74
else:
    st.info("Add a plot with a location in **My Farm** for location-specific forecasts. Using a default location for now.")
    lat, lon = 19.07, 74.74

fc = get_forecast_with_meta(lat, lon)
forecast = fc.records
assessments = assess_forecast(forecast)

st.caption(live_data_caption(fc.source, fc.fetched_at))

df = pd.DataFrame(
    {
        "Date": [w.day for w in forecast],
        "Temp (°C)": [w.temperature_c for w in forecast],
        "Humidity (%)": [w.humidity_percent for w in forecast],
        "Rain chance (%)": [w.rain_chance_percent for w in forecast],
        "Wind (km/h)": [w.wind_speed_kmh for w in forecast],
    }
).set_index("Date")

st.subheader("7-day forecast")
st.line_chart(df[["Temp (°C)", "Humidity (%)"]])
st.bar_chart(df[["Rain chance (%)", "Wind (km/h)"]])

st.subheader("Daily disease risk & spray guidance")
for w, a in zip(forecast, assessments):
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
        c1.markdown(f"**{w.day.strftime('%a %d %b')}**")
        c1.caption(f"{w.temperature_c:.0f}°C · {w.humidity_percent:.0f}% humidity · {w.condition}")
        c2.markdown("Mildew")
        c2.markdown(risk_badge(a.powdery_mildew_risk))
        c3.markdown("Rust")
        c3.markdown(risk_badge(a.leaf_rust_risk))
        c4.markdown("Fungal")
        c4.markdown(risk_badge(a.fungal_disease_risk))
        c5.markdown("Pests")
        c5.markdown(risk_badge(a.pest_outbreak_risk))
        st.caption(("✅ " if a.spray_recommended else "🚫 ") + a.best_spray_window_note)
        st.caption(f"💧 {a.irrigation_suggestion}")
        if a.heat_stress_warning:
            st.warning("Heat stress warning")
