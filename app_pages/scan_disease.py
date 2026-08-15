import json
from pathlib import Path

import streamlit as st
from PIL import Image

from ai.classifier import AI_ADVISORY_DISCLAIMER, diagnose, model_metadata
from core import auth, db
from core.helpers import add_notification, confidence_badge, plot_options, risk_badge

user = auth.require_login()
owner_id = user["id"]

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "diseases_knowledge_base.json"
KNOWLEDGE_BASE = {e["label_key"]: e for e in json.loads(KB_PATH.read_text(encoding="utf-8"))}

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "sample_leaves"
SAMPLE_IMAGES = {
    "Healthy": ["healthy_1.jpg", "healthy_2.jpg", "healthy_3.jpg"],
    "Leaf Rust": ["leaf_rust_1.jpg", "leaf_rust_2.jpg", "leaf_rust_3.jpg"],
    "Leaf Spot": ["leaf_spot_1.jpg", "leaf_spot_2.jpg", "leaf_spot_3.jpg"],
}

st.title("🔬 Scan Disease")
st.caption("Upload or capture a close-up mulberry leaf photo. The on-device-style AI model runs entirely on this server.")

meta = model_metadata()
with st.expander("About this AI model", icon=":material/info:"):
    st.write(
        "This scan uses **two models together**:\n\n"
        "1. **MobileNetV3-Small**, fine-tuned on 1,091 real DSLR photos "
        "(Kaggle `nahiduzzaman13/mulberry-leaf-dataset`, CC0) — **96.7% test accuracy**. "
        "Used whenever the leaf looks healthy, rust-affected, or spot-affected, since "
        "those are the only 3 of the app's 25 tracked conditions with real labeled "
        "photos available. Compared against 10 other model/feature combinations "
        "(hand-crafted features and frozen MobileNet embeddings, across 5 classical "
        "algorithms each) — it won outright; see `data/mobilenet_training_results.json` "
        "for the full comparison.\n\n"
        f"2. **{meta.get('algorithm', 'RandomForestClassifier (scikit-learn)')}** "
        f"covering all {meta.get('class_count', 25)} conditions (pests, deficiencies, "
        f"stress, and the diseases above) — falls back to this for anything outside "
        f"the 3 real-photo-trained classes. Held-out accuracy on synthetic validation "
        f"data: {meta.get('validation_accuracy', 0) * 100:.1f}%.\n\n"
        f"{meta.get('notes', '')}"
    )

plots = plot_options(owner_id)
plot_id = None
if plots:
    plot_id = st.selectbox("Plot (optional)", options=[""] + list(plots.keys()), format_func=lambda k: "— none —" if k == "" else plots[k])

scan_type = st.radio("Scan type", ["Leaf", "Whole plant", "Stem", "Root area"], horizontal=True)

source = st.radio("Image source", ["Try a sample", "Upload photo", "Use camera"], horizontal=True)

image = None
image_bytes = None

if source == "Use camera":
    image_file = st.camera_input("Capture leaf photo")
    if image_file is not None:
        image_bytes = image_file.getvalue()
        image = Image.open(image_file)

elif source == "Upload photo":
    image_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"])
    if image_file is not None:
        image_bytes = image_file.getvalue()
        image = Image.open(image_file)

else:  # Try a sample
    st.caption(
        "Real photos from the training dataset (Kaggle `nahiduzzaman13/mulberry-leaf-dataset`) — "
        "pick one to see the real-photo-trained model in action without needing your own photo."
    )
    for cls_label, filenames in SAMPLE_IMAGES.items():
        st.markdown(f"**{cls_label}**")
        cols = st.columns(len(filenames))
        for col, fname in zip(cols, filenames):
            with col:
                st.image(str(SAMPLE_DIR / fname), width="stretch")
                if st.button("Use this photo", key=f"sample_{fname}", width="stretch"):
                    st.session_state["scan_sample_choice"] = fname
                    st.rerun()

    chosen = st.session_state.get("scan_sample_choice")
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
        st.subheader(result.name_en)
        st.caption(f"{result.name_mr} · {result.name_hi}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{result.confidence * 100:.0f}%")
        m2.metric("Severity", f"{result.severity_percent:.0f}%")
        m3.markdown("**Risk**")
        m3.markdown(risk_badge(result.risk_level))
        st.markdown(f"Confidence tier: {confidence_badge(result.confidence_tier)}")
        if result.source == "real_photo_trained":
            st.caption("🎯 Diagnosed by the MobileNetV3-Small model, fine-tuned on 1,091 real leaf photos (96.7% test accuracy).")
        else:
            st.caption("🧪 Diagnosed by the synthetic-feature model — no real labeled photos exist yet for this condition.")

        if result.top_alternatives:
            alt_text = ", ".join(f"{k.replace('_', ' ')} ({p * 100:.0f}%)" for k, p in result.top_alternatives)
            st.caption(f"Other possibilities: {alt_text}")

        st.info(AI_ADVISORY_DISCLAIMER, icon=":material/warning:")

        kb = KNOWLEDGE_BASE.get(result.label_key)
        if kb and kb["category"] != "healthy":
            with st.expander("Symptoms & cause", expanded=True):
                st.write(f"**Symptoms:** {kb['symptoms']}")
                st.write(f"**Cause:** {kb['cause']}")
                st.write(f"**Favorable weather:** {kb['favorable_weather']}")
            with st.expander("Recommended treatment"):
                st.write(f"**Organic control:** {kb['organic_control']}")
                st.write(f"**Chemical control:** {kb['chemical_control']}")
                st.write(f"**Cultural practice:** {kb['cultural_practice']}")
                st.write(f"**Prevention:** {kb['prevention']}")
                st.caption(f"Safety interval: {kb['safety_interval_days']} days · Re-entry: {kb['re_entry_interval_hours']} hours")
            if result.confidence < 0.5 or kb.get("expert_escalation_condition"):
                st.warning(f"Escalate to an expert if: {kb.get('expert_escalation_condition')}")
        elif kb:
            st.success("This leaf looks healthy. Keep up regular monitoring and balanced nutrition.")

        if st.button("Save this scan to my records", type="primary", icon=":material/save:"):
            image_path = db.save_uploaded_image(image_bytes, prefix="scan")
            db.insert_row(
                "scan_record",
                {
                    "owner_id": owner_id,
                    "plot_id": plot_id or None,
                    "scan_type": scan_type.lower().replace(" ", "_"),
                    "image_path": image_path,
                    "captured_at": db.now_iso(),
                    "result_label_key": result.label_key,
                    "display_name": result.name_en,
                    "confidence": result.confidence,
                    "severity_percent": result.severity_percent,
                    "risk_level": result.risk_level,
                    "model_version": result.model_version,
                    "model_accuracy": result.model_accuracy,
                    "used_cloud_fallback": 0,
                },
            )
            if result.risk_level in ("high", "severe"):
                add_notification(
                    owner_id, "scan_alert", f"High risk: {result.name_en}",
                    f"A recent scan detected {result.name_en} at {result.severity_percent:.0f}% severity. Consider expert review.",
                )
            st.success("Scan saved. View it on the Dashboard or Scan History.")

st.divider()
st.subheader("Scan history")
history = db.fetch_all("scan_record", "owner_id = ?", (owner_id,), order_by="captured_at DESC")
if not history:
    st.caption("No scans saved yet.")
else:
    for scan in history[:20]:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.markdown(f"**{scan['display_name'] or scan['result_label_key']}**")
            c1.caption(scan["captured_at"][:16].replace("T", " "))
            c2.markdown(f"Confidence: {scan['confidence'] * 100:.0f}%")
            c3.markdown(risk_badge(scan["risk_level"] or "low"))
            if c4.button("Delete", key=f"del_scan_{scan['id']}", icon=":material/delete:"):
                db.delete_row("scan_record", scan["id"])
                st.rerun()
