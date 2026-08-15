import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from core import auth, db
from core.geocoding import find_location

user = auth.require_login()
owner_id = user["id"]

st.title("🌱 My Farm")
st.caption("Manage your mulberry plots. Each plot can be linked to scans, logs and production records.")

with st.expander("➕ Add a new plot", expanded=not db.count("farm_plot", "owner_id = ?", (owner_id,))):
    st.markdown("**📍 Set the plot's location** (used for weather & disease-risk forecasts)")
    loc_col1, loc_col2 = st.columns([1, 1.6], gap="medium")

    with loc_col1:
        st.caption("Use your device's GPS:")
        location = streamlit_geolocation()
        if location and location.get("latitude") is not None:
            st.session_state["new_plot_lat"] = round(location["latitude"], 4)
            st.session_state["new_plot_lon"] = round(location["longitude"], 4)
            st.success(f"Detected: {location['latitude']:.4f}, {location['longitude']:.4f}")

    with loc_col2:
        st.caption("Or find it by PIN code / city name:")
        s1, s2 = st.columns([3, 1])
        query = s1.text_input(
            "PIN code or city", label_visibility="collapsed",
            placeholder="e.g. 414001 or Pune", key="location_search_query",
        )
        if s2.button("Find", key="location_search_btn", width="stretch"):
            if query.strip():
                results = find_location(query)
                st.session_state["location_search_results"] = results
                if not results:
                    st.warning("No matches found. Try a different PIN code or city name.")
            else:
                st.session_state["location_search_results"] = []

        results = st.session_state.get("location_search_results") or []
        if results:
            options = {i: r["label"] for i, r in enumerate(results)}
            picked = st.selectbox(
                "Matches", options=list(options.keys()), format_func=lambda i: options[i],
                key="location_search_pick", label_visibility="collapsed",
            )
            if st.button("Use this location", key="use_searched_location", icon=":material/pin_drop:"):
                chosen = results[picked]
                st.session_state["new_plot_lat"] = round(chosen["latitude"], 4)
                st.session_state["new_plot_lon"] = round(chosen["longitude"], 4)
                st.session_state["location_search_results"] = []
                st.rerun()

    st.divider()

    with st.form("add_plot_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Plot name*", placeholder="e.g. North field")
        area = c2.number_input("Area (acres)*", min_value=0.0, step=0.1)
        c3, c4 = st.columns(2)
        variety = c3.text_input("Mulberry variety", placeholder="e.g. V1, S-1635")
        irrigation = c4.selectbox("Irrigation type", ["Drip", "Sprinkler", "Flood", "Rainfed", "Other"])
        c5, c6 = st.columns(2)
        soil_type = c5.selectbox("Soil type", ["Red", "Black", "Loamy", "Sandy", "Clay", "Other"])
        plantation_date = c6.date_input("Plantation date", value=None)
        c7, c8 = st.columns(2)
        latitude = c7.number_input(
            "Latitude", value=st.session_state.get("new_plot_lat", 19.07), format="%.4f", key="new_plot_lat",
        )
        longitude = c8.number_input(
            "Longitude", value=st.session_state.get("new_plot_lon", 74.74), format="%.4f", key="new_plot_lon",
        )
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save plot", type="primary")

    if submitted:
        if not name.strip() or area <= 0:
            st.error("Plot name and area are required.")
        else:
            db.insert_row(
                "farm_plot",
                {
                    "owner_id": owner_id,
                    "name": name.strip(),
                    "area_acres": area,
                    "mulberry_variety": variety,
                    "plantation_date": plantation_date.isoformat() if plantation_date else "",
                    "irrigation_type": irrigation,
                    "soil_type": soil_type,
                    "latitude": latitude,
                    "longitude": longitude,
                    "notes": notes,
                    "photo_paths": "",
                    "health_score": 100,
                },
            )
            st.success(f"Plot '{name}' added.")
            st.rerun()

st.divider()

plots = db.fetch_all("farm_plot", "owner_id = ?", (owner_id,), order_by="created_at DESC")
if not plots:
    st.info("No plots yet — add your first one above.")
else:
    for plot in plots:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.markdown(f"**{plot['name']}**")
            c1.caption(f"{plot['mulberry_variety'] or 'Variety not set'} · {plot['soil_type'] or 'Soil n/a'} · {plot['irrigation_type'] or 'Irrigation n/a'}")
            c2.metric("Area", f"{plot['area_acres']} ac")
            c3.metric("Health score", plot["health_score"] if plot["health_score"] is not None else "—")
            scan_count = db.count("scan_record", "plot_id = ?", (plot["id"],))
            c4.metric("Scans", scan_count)
            if plot["latitude"] and plot["longitude"]:
                c1.caption(f"📍 {plot['latitude']:.4f}, {plot['longitude']:.4f}")
            if plot["notes"]:
                st.caption(f"📝 {plot['notes']}")
            del_col, _ = st.columns([1, 5])
            if del_col.button("Delete plot", key=f"del_{plot['id']}", icon=":material/delete:"):
                db.delete_row("farm_plot", plot["id"])
                st.rerun()
