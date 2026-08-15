import json
from pathlib import Path

import streamlit as st
from PIL import Image

from ai.alert_engine import generate_disease_alert
from ai.silkworm_classifier import AI_ADVISORY_DISCLAIMER, diagnose, is_available, model_metadata
from core import auth, db
from core.helpers import plot_options, risk_badge

user = auth.require_login()
owner_id = user["id"]

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "silkworm_knowledge_base.json"
KNOWLEDGE_BASE = {e["label_key"]: e for e in json.loads(KB_PATH.read_text(encoding="utf-8"))}

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "sample_silkworms"
SAMPLE_IMAGES = {
    "Healthy": ["healthy_1.jpg", "healthy_2.jpg", "healthy_3.jpg"],
    "Grasserie": ["grasserie_1.jpg", "grasserie_2.jpg", "grasserie_3.jpg"],
    "Infected (other)": ["infected_1.jpg", "infected_2.jpg", "infected_3.jpg"],
}

st.title("🐛 Scan Silkworm")
st.caption("Upload or capture a close-up photo of a silkworm larva. The AI model runs entirely on this server.")

if not is_available():
    st.warning("The silkworm model hasn't been trained yet. Run `python -m training.train_silkworm_compare` first.")
    st.stop()

meta = model_metadata()
best = meta.get("results", [{}])[0] if meta.get("results") else {}
with st.expander("About this AI model", icon=":material/info:"):
    st.write(
        "**Frozen MobileNetV3-Small embeddings + SVM (RBF)**, trained on **4,862 real photos** "
        "(Kaggle `tatinanikhitha/silkworm`, Andhra Pradesh sericulture field data) — "
        f"**{best.get('accuracy', 0) * 100:.1f}% test accuracy**. This won a head-to-head "
        "comparison against 10 other model/feature combinations (hand-crafted features and "
        "a fully fine-tuned MobileNetV3-Small, across 5 classical algorithms each) — see "
        "`data/silkworm_training_results.json` for the full comparison.\n\n"
        "**Honesty note:** this dataset names one specific disease — **Grasserie** — but its "
        "other diseased photos are only labeled generically as \"Infected,\" not broken down into "
        "Flacherie, Muscardine, Pébrine etc. individually. A result of **\"Signs of illness "
        "(unconfirmed cause)\"** means the model is confident something is wrong, not which "
        "specific disease it is — always use Submit to Expert for those."
    )

plots = plot_options(owner_id)
plot_id = None
if plots:
    plot_id = st.selectbox("Plot (optional)", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k])

source = st.radio("Image source", ["Try a sample", "Upload photo", "Use camera"], horizontal=True)

image = None
image_bytes = None

if source == "Use camera":
    image_file = st.camera_input("Capture silkworm photo")
    if image_file is not None:
        image_bytes = image_file.getvalue()
        image = Image.open(image_file)

elif source == "Upload photo":
    image_file = st.file_uploader("Upload a silkworm photo", type=["jpg", "jpeg", "png", "webp"])
    if image_file is not None:
        image_bytes = image_file.getvalue()
        image = Image.open(image_file)

else:  # Try a sample
    st.caption(
        "Real photos from the training dataset (Kaggle `tatinanikhitha/silkworm`) — "
        "pick one to see the real-photo-trained model in action without needing your own photo."
    )
    for cls_label, filenames in SAMPLE_IMAGES.items():
        st.markdown(f"**{cls_label}**")
        cols = st.columns(len(filenames))
        for col, fname in zip(cols, filenames):
            with col:
                st.image(str(SAMPLE_DIR / fname), width="stretch")
                if st.button("Use this photo", key=f"silkworm_sample_{fname}", width="stretch"):
                    st.session_state["silkworm_scan_sample_choice"] = fname
                    st.rerun()

    chosen = st.session_state.get("silkworm_scan_sample_choice")
    if chosen:
        sample_path = SAMPLE_DIR / chosen
        image_bytes = sample_path.read_bytes()
        image = Image.open(sample_path)
        st.info(f"Selected sample: **{chosen}**", icon=":material/check_circle:")

if image is not None:
    col_img, col_result = st.columns([1, 1.3], gap="large")
    with col_img:
        st.image(image, caption="Photo being analyzed", width="stretch")

    with st.spinner("Running AI inference..."):
        result = diagnose(image)

    with col_result:
        kb = KNOWLEDGE_BASE.get(result.label_key)

        if result.is_inconclusive:
            st.subheader("Inconclusive")
            st.caption(f"Top guess: {kb['name_en'] if kb else result.label_key} at only {result.confidence * 100:.0f}% confidence")
            st.info("The model isn't confident enough to give a reliable result. Try a clearer, closer photo, or use Submit to Expert.")
        else:
            st.subheader(kb["name_en"] if kb else result.label_key)
            st.caption(f"{kb['name_mr']} · {kb['name_hi']}" if kb else "")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{result.confidence * 100:.0f}%")
            m2.markdown("**Risk**")
            m2.markdown(risk_badge(result.risk_level))
            st.caption(f"🎯 Diagnosed by the MobileNetV3-Small embedding + SVM silkworm model ({best.get('accuracy', 0) * 100:.1f}% test accuracy).")

            if result.top_alternatives:
                alt_text = ", ".join(f"{KNOWLEDGE_BASE.get(k, {}).get('name_en', k)} ({p * 100:.0f}%)" for k, p in result.top_alternatives)
                st.caption(f"Other possibilities: {alt_text}")

            st.info(AI_ADVISORY_DISCLAIMER, icon=":material/warning:")

            if kb and kb["category"] != "healthy":
                with st.expander("Symptoms & cause", expanded=True):
                    st.write(f"**Symptoms:** {kb['symptoms']}")
                    st.write(f"**Cause:** {kb['cause']}")
                    st.write(f"**Favorable conditions:** {kb['favorable_weather']}")
                with st.expander("Recommended response"):
                    st.write(f"**Organic control:** {kb['organic_control']}")
                    st.write(f"**Chemical control:** {kb['chemical_control']}")
                    st.write(f"**Cultural practice:** {kb['cultural_practice']}")
                    st.write(f"**Prevention:** {kb['prevention']}")
                st.warning(f"Escalate to an expert if: {kb.get('expert_escalation_condition')}")
            elif kb:
                st.success("This silkworm looks healthy. Keep up regular monitoring and bed hygiene.")

            if st.button("Save this scan to my records", type="primary", icon=":material/save:"):
                image_path = db.save_uploaded_image(image_bytes, prefix="silkworm_scan")
                db.insert_row(
                    "scan_record",
                    {
                        "owner_id": owner_id,
                        "plot_id": plot_id or None,
                        "scan_type": "silkworm",
                        "image_path": image_path,
                        "captured_at": db.now_iso(),
                        "result_label_key": result.label_key,
                        "display_name": kb["name_en"] if kb else result.label_key,
                        "confidence": result.confidence,
                        "severity_percent": 0.0,
                        "risk_level": result.risk_level,
                        "model_version": result.model_version,
                        "model_accuracy": result.model_accuracy,
                        "used_cloud_fallback": 0,
                    },
                )
                generate_disease_alert(owner_id, kb["name_en"] if kb else result.label_key, result.risk_level, 0.0, "silkworm")
                st.success("Scan saved. View it on the Dashboard or Scan History.")

st.divider()
st.subheader("Scan history")
history = db.fetch_all("scan_record", "owner_id = ? AND scan_type = 'silkworm'", (owner_id,), order_by="captured_at DESC")
if not history:
    st.caption("No silkworm scans saved yet.")
else:
    for scan in history[:20]:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.markdown(f"**{scan['display_name'] or scan['result_label_key']}**")
            c1.caption(scan["captured_at"][:16].replace("T", " "))
            c2.markdown(f"Confidence: {scan['confidence'] * 100:.0f}%")
            c3.markdown(risk_badge(scan["risk_level"] or "low"))
            if c4.button("Delete", key=f"del_silkworm_scan_{scan['id']}", icon=":material/delete:"):
                db.delete_row("scan_record", scan["id"])
                st.rerun()
