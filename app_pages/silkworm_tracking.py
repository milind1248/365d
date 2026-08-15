import datetime as dt

import streamlit as st

from ai.alert_engine import run_batch_checks
from core import auth, db
from core.helpers import plot_options

user = auth.require_login()
owner_id = user["id"]

INSTAR_STAGES = ["1st", "2nd", "3rd", "4th", "5th", "Spinning"]

st.title("🐛 Silkworm Rearing Tracker")
st.caption(
    "Log your rearing batches (instar stage, feeding, mortality) so the Alert Engine can remind you "
    "when feeding is due, a molt is approaching, or mortality needs attention."
)

plots = plot_options(owner_id)

with st.expander("➕ Add a new batch", expanded=not db.count("silkworm_batch", "owner_id = ?", (owner_id,))):
    with st.form("add_batch_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        batch_name = c1.text_input("Batch / tray name*", placeholder="e.g. Tray A")
        instar_stage = c2.selectbox("Current instar stage", INSTAR_STAGES)
        c3, c4 = st.columns(2)
        plot_id = c3.selectbox("Plot (optional)", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k])
        mortality_percent = c4.number_input("Mortality so far (%)", min_value=0.0, max_value=100.0, step=0.1)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add batch", type="primary")

    if submitted:
        if not batch_name.strip():
            st.error("Batch name is required.")
        else:
            now = db.now_iso()
            db.insert_row(
                "silkworm_batch",
                {
                    "owner_id": owner_id, "plot_id": plot_id or None, "batch_name": batch_name.strip(),
                    "instar_stage": instar_stage, "mortality_percent": mortality_percent,
                    "last_fed_at": now, "updated_at": now, "notes": notes,
                },
            )
            st.success(f"Batch '{batch_name}' added.")
            st.rerun()

st.divider()
st.subheader("Active batches")

batches = db.fetch_all("silkworm_batch", "owner_id = ?", (owner_id,), order_by="created_at DESC")
if not batches:
    st.info("No batches yet — add your first one above.")
else:
    for batch in batches:
        run_batch_checks(owner_id, batch)

        with st.container(border=True):
            c1, c2, c3 = st.columns([2.2, 1, 1])
            c1.markdown(f"**{batch['batch_name']}**")
            c1.caption(f"Instar: {batch['instar_stage']}" + (f" · {plots.get(batch['plot_id'], '')}" if batch["plot_id"] else ""))
            if batch["last_fed_at"]:
                last_fed = dt.datetime.fromisoformat(batch["last_fed_at"])
                hours_ago = (dt.datetime.now(dt.timezone.utc) - last_fed).total_seconds() / 3600
                c1.caption(f"🍃 Last fed {hours_ago:.0f}h ago")
            c2.metric("Mortality", f"{batch['mortality_percent'] or 0:.1f}%")

            if c3.button("Mark fed now", key=f"feed_{batch['id']}", icon=":material/restaurant:", width="stretch"):
                db.update_row("silkworm_batch", batch["id"], {"last_fed_at": db.now_iso()})
                st.rerun()

            with st.form(f"update_form_{batch['id']}", border=False):
                u1, u2, u3 = st.columns([1.5, 1, 1])
                new_stage = u1.selectbox("Update instar", INSTAR_STAGES, index=INSTAR_STAGES.index(batch["instar_stage"]) if batch["instar_stage"] in INSTAR_STAGES else 0, key=f"stage_{batch['id']}")
                new_mortality = u2.number_input("Mortality (%)", min_value=0.0, max_value=100.0, step=0.1, value=float(batch["mortality_percent"] or 0), key=f"mort_{batch['id']}")
                update_submitted = u3.form_submit_button("Update", icon=":material/save:")
            if update_submitted:
                fields = {"updated_at": db.now_iso(), "mortality_percent": new_mortality}
                if new_stage != batch["instar_stage"]:
                    fields["instar_stage"] = new_stage
                db.update_row("silkworm_batch", batch["id"], fields)
                st.success("Batch updated.")
                st.rerun()

            if batch["notes"]:
                st.caption(f"📝 {batch['notes']}")

            if st.button("Delete batch", key=f"del_batch_{batch['id']}", icon=":material/delete:"):
                db.delete_row("silkworm_batch", batch["id"])
                st.rerun()
