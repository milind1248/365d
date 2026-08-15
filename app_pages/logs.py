import datetime as dt

import streamlit as st

from core import auth, db
from core.helpers import plot_options

user = auth.require_login()
owner_id = user["id"]

st.title("🧪 Spray, Fertilizer & Soil Logs")
st.caption("Keep a record of every treatment applied and soil test taken, per plot.")

plots = plot_options(owner_id)
tab_spray, tab_fert, tab_soil = st.tabs(["Spray log", "Fertilizer log", "Soil test"])

# ---------------------------------------------------------------- Spray log
with tab_spray:
    with st.expander("➕ Add spray record", expanded=True):
        with st.form("spray_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            date = c1.date_input("Date", value=dt.date.today(), key="spray_date")
            plot_id = c2.selectbox("Plot", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k], key="spray_plot")
            c3, c4 = st.columns(2)
            problem = c3.text_input("Problem treated", placeholder="e.g. Powdery mildew")
            product = c4.text_input("Product used", placeholder="e.g. Neem oil 5ml/L")
            c5, c6, c7 = st.columns(3)
            is_organic = c5.checkbox("Organic", value=True)
            water_qty = c6.number_input("Water (litres)", min_value=0.0, step=1.0)
            area_covered = c7.number_input("Area covered (acres)", min_value=0.0, step=0.1)
            c8, c9 = st.columns(2)
            labor_cost = c8.number_input("Labor cost (₹)", min_value=0.0, step=10.0)
            follow_up = c9.date_input("Follow-up date", value=None)
            dosage = st.text_input("Dosage")
            notes = st.text_area("Notes", key="spray_notes")
            submitted = st.form_submit_button("Save spray log", type="primary")
        if submitted:
            db.insert_row(
                "spray_log",
                {
                    "owner_id": owner_id, "plot_id": plot_id or None, "date": date.isoformat(),
                    "problem_treated": problem, "product_used": product, "is_organic": int(is_organic),
                    "dosage": dosage, "water_quantity_litres": water_qty, "area_covered_acres": area_covered,
                    "labor_cost": labor_cost, "notes": notes,
                    "follow_up_date": follow_up.isoformat() if follow_up else "",
                },
            )
            st.success("Spray log saved.")
            st.rerun()

    for row in db.fetch_all("spray_log", "owner_id = ?", (owner_id,), order_by="date DESC"):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            tag = "🌿 Organic" if row["is_organic"] else "🧪 Chemical"
            c1.markdown(f"**{row['problem_treated'] or 'Spray'}** — {row['product_used']} ({tag})")
            c1.caption(f"{row['date']} · {row['dosage'] or ''} · ₹{row['labor_cost'] or 0} labor")
            if c2.button("Delete", key=f"del_spray_{row['id']}", icon=":material/delete:"):
                db.delete_row("spray_log", row["id"])
                st.rerun()

# ------------------------------------------------------------ Fertilizer log
with tab_fert:
    with st.expander("➕ Add fertilizer record", expanded=True):
        with st.form("fert_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            date = c1.date_input("Date", value=dt.date.today(), key="fert_date")
            plot_id = c2.selectbox("Plot", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k], key="fert_plot")
            c3, c4 = st.columns(2)
            fert_name = c3.text_input("Fertilizer name", placeholder="e.g. Urea, FYM, NPK 19:19:19")
            quantity = c4.number_input("Quantity (kg)", min_value=0.0, step=1.0)
            is_organic = st.checkbox("Organic", value=False, key="fert_organic")
            notes = st.text_area("Notes", key="fert_notes")
            submitted = st.form_submit_button("Save fertilizer log", type="primary")
        if submitted:
            db.insert_row(
                "fertilizer_log",
                {
                    "owner_id": owner_id, "plot_id": plot_id or None, "date": date.isoformat(),
                    "fertilizer_name": fert_name, "quantity_kg": quantity,
                    "is_organic": int(is_organic), "notes": notes,
                },
            )
            st.success("Fertilizer log saved.")
            st.rerun()

    for row in db.fetch_all("fertilizer_log", "owner_id = ?", (owner_id,), order_by="date DESC"):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            tag = "🌿 Organic" if row["is_organic"] else "🧪 Inorganic"
            c1.markdown(f"**{row['fertilizer_name']}** — {row['quantity_kg']} kg ({tag})")
            c1.caption(row["date"])
            if c2.button("Delete", key=f"del_fert_{row['id']}", icon=":material/delete:"):
                db.delete_row("fertilizer_log", row["id"])
                st.rerun()

# ---------------------------------------------------------------- Soil test
with tab_soil:
    with st.expander("➕ Add soil test", expanded=True):
        with st.form("soil_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            test_date = c1.date_input("Test date", value=dt.date.today(), key="soil_date")
            plot_id = c2.selectbox("Plot", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k], key="soil_plot")
            c3, c4 = st.columns(2)
            ph = c3.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
            ec = c4.number_input("EC (dS/m)", min_value=0.0, step=0.1)
            c5, c6, c7 = st.columns(3)
            n_level = c5.selectbox("Nitrogen", ["Low", "Medium", "High"], key="n_level")
            p_level = c6.selectbox("Phosphorus", ["Low", "Medium", "High"], key="p_level")
            k_level = c7.selectbox("Potassium", ["Low", "Medium", "High"], key="k_level")
            organic_carbon = st.number_input("Organic carbon (%)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Save soil test", type="primary")
        if submitted:
            db.insert_row(
                "soil_test",
                {
                    "owner_id": owner_id, "plot_id": plot_id or None, "test_date": test_date.isoformat(),
                    "ph": ph, "ec": ec, "nitrogen_level": n_level, "phosphorus_level": p_level,
                    "potassium_level": k_level, "organic_carbon_percent": organic_carbon,
                },
            )
            st.success("Soil test saved.")
            st.rerun()

    for row in db.fetch_all("soil_test", "owner_id = ?", (owner_id,), order_by="test_date DESC"):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**pH {row['ph']}** · EC {row['ec']} · N:{row['nitrogen_level']} P:{row['phosphorus_level']} K:{row['potassium_level']}")
            c1.caption(f"{row['test_date']} · Organic carbon {row['organic_carbon_percent']}%")
            if c2.button("Delete", key=f"del_soil_{row['id']}", icon=":material/delete:"):
                db.delete_row("soil_test", row["id"])
                st.rerun()
