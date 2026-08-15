import datetime as dt

import pandas as pd
import streamlit as st

from core import auth, db
from core.helpers import plot_options

user = auth.require_login()
owner_id = user["id"]

st.title("📦 Production Records")
st.caption("Track leaf harvest, cocoon production and sales per plot.")

plots = plot_options(owner_id)

with st.expander("➕ Add production record", expanded=True):
    with st.form("prod_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        date = c1.date_input("Date", value=dt.date.today())
        plot_id = c2.selectbox("Plot", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k])
        c3, c4 = st.columns(2)
        leaf_kg = c3.number_input("Leaf harvest (kg)", min_value=0.0, step=1.0)
        cocoon_kg = c4.number_input("Cocoon production (kg)", min_value=0.0, step=1.0)
        c5, c6 = st.columns(2)
        sale_qty = c5.number_input("Sale quantity (kg)", min_value=0.0, step=1.0)
        sale_rate = c6.number_input("Sale rate (₹/kg)", min_value=0.0, step=1.0)
        c7, c8 = st.columns(2)
        buyer = c7.text_input("Buyer")
        expenses = c8.number_input("Expenses (₹)", min_value=0.0, step=10.0)
        submitted = st.form_submit_button("Save record", type="primary")
    if submitted:
        db.insert_row(
            "production_record",
            {
                "owner_id": owner_id, "plot_id": plot_id or None, "date": date.isoformat(),
                "leaf_harvest_kg": leaf_kg, "cocoon_production_kg": cocoon_kg,
                "sale_quantity_kg": sale_qty, "sale_rate_per_kg": sale_rate,
                "buyer": buyer, "expenses": expenses,
            },
        )
        st.success("Production record saved.")
        st.rerun()

st.divider()

records = db.fetch_all("production_record", "owner_id = ?", (owner_id,), order_by="date ASC")
if not records:
    st.info("No production records yet.")
else:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["sale_quantity_kg"] * df["sale_rate_per_kg"]
    df["net"] = df["revenue"] - df["expenses"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total leaf harvest", f"{df['leaf_harvest_kg'].sum():,.0f} kg")
    m2.metric("Total cocoon", f"{df['cocoon_production_kg'].sum():,.0f} kg")
    m3.metric("Total revenue", f"₹{df['revenue'].sum():,.0f}")
    m4.metric("Net profit", f"₹{df['net'].sum():,.0f}")

    st.subheader("Harvest over time")
    chart_df = df.set_index("date")[["leaf_harvest_kg", "cocoon_production_kg"]]
    st.line_chart(chart_df)

    st.subheader("Revenue vs expenses")
    st.bar_chart(df.set_index("date")[["revenue", "expenses"]])

    st.subheader("All records")
    display_cols = ["date", "leaf_harvest_kg", "cocoon_production_kg", "sale_quantity_kg", "sale_rate_per_kg", "buyer", "revenue", "expenses", "net"]
    st.dataframe(df[display_cols].sort_values("date", ascending=False), width="stretch", hide_index=True)
