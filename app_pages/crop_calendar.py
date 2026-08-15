import datetime as dt

import streamlit as st

from core import auth, db
from core.helpers import plot_options

user = auth.require_login()
owner_id = user["id"]

st.title("📅 Crop Calendar")
st.caption("Track spraying, fertilizing, pruning and harvest tasks across your plots.")

plots = plot_options(owner_id)

with st.expander("➕ Add a task", expanded=True):
    with st.form("add_task_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        title = c1.text_input("Task title*", placeholder="e.g. Apply neem spray")
        category = c2.selectbox("Category", ["Spraying", "Fertilizing", "Pruning", "Irrigation", "Harvest", "Soil test", "Other"])
        c3, c4 = st.columns(2)
        due_date = c3.date_input("Due date", value=dt.date.today())
        plot_id = c4.selectbox("Plot", options=[""] + list(plots.keys()), format_func=lambda k: "— any plot —" if k == "" else plots[k])
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add task", type="primary")

    if submitted:
        if not title.strip():
            st.error("Task title is required.")
        else:
            db.insert_row(
                "crop_task",
                {
                    "owner_id": owner_id, "plot_id": plot_id or None, "title": title.strip(),
                    "category": category, "due_date": due_date.isoformat(),
                    "is_completed": 0, "notes": notes,
                },
            )
            st.success("Task added.")
            st.rerun()

st.divider()

tab_pending, tab_done = st.tabs(["Pending", "Completed"])

with tab_pending:
    tasks = db.fetch_all("crop_task", "owner_id = ? AND is_completed = 0", (owner_id,), order_by="due_date ASC")
    if not tasks:
        st.info("No pending tasks.")
    for task in tasks:
        overdue = task["due_date"] and task["due_date"] < dt.date.today().isoformat()
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            label = f"**{task['title']}**" + (" 🔴 overdue" if overdue else "")
            c1.markdown(label)
            c1.caption(f"{task['category']} · due {task['due_date']}" + (f" · {plots.get(task['plot_id'], '')}" if task["plot_id"] else ""))
            if task["notes"]:
                c1.caption(f"📝 {task['notes']}")
            if c2.button("Mark done", key=f"done_{task['id']}", icon=":material/check:"):
                db.update_row("crop_task", task["id"], {"is_completed": 1})
                st.rerun()
            if c3.button("Delete", key=f"del_task_{task['id']}", icon=":material/delete:"):
                db.delete_row("crop_task", task["id"])
                st.rerun()

with tab_done:
    done_tasks = db.fetch_all("crop_task", "owner_id = ? AND is_completed = 1", (owner_id,), order_by="due_date DESC")
    if not done_tasks:
        st.caption("No completed tasks yet.")
    for task in done_tasks:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"~~{task['title']}~~")
            c1.caption(f"{task['category']} · was due {task['due_date']}")
            if c2.button("Delete", key=f"del_done_{task['id']}", icon=":material/delete:"):
                db.delete_row("crop_task", task["id"])
                st.rerun()
