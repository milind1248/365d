import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core import auth

auth.require_login()

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "mobilenet_training_results.json"
data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

class_names = data["class_names"]
short_names = {"Disease Free leaves": "Healthy", "Leaf Rust": "Leaf Rust", "Leaf spot": "Leaf Spot"}
results = data["results"]
winner = results[0]

FSET_LABEL = {
    "mobilenet_finetuned": "MobileNet · fine-tuned",
    "mobilenet_embed": "MobileNet embeddings + classical ML",
    "hand_crafted": "Hand-crafted features + classical ML",
}
FSET_COLOR = {
    "mobilenet_finetuned": "#4C9A2A",
    "mobilenet_embed": "#3B82C4",
    "hand_crafted": "#C08A2E",
}

st.title("📊 AI Model Report")
st.caption(
    "How the leaf-disease classifier was chosen: 11 model/feature combinations, trained and "
    "tested on the same 1,091 real photos from the Kaggle mulberry-leaf-dataset (CC0)."
)

# ---------------------------------------------------------------- Hero stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Real photos", data["train_size"] + data["test_size"])
c2.metric("Train / test split", f"{data['train_size']} / {data['test_size']}")
c3.metric("Combinations tried", len(results))
c4.metric("Best accuracy", f"{winner['accuracy'] * 100:.1f}%", winner["algorithm"])

st.success(
    f"**Winner: {winner['algorithm']}** ({FSET_LABEL[winner['feature_set']]}) — "
    f"{winner['accuracy'] * 100:.1f}% test accuracy, {winner['f1_macro']:.3f} macro-F1. "
    "This is the model currently running on the **Scan Disease** page for healthy / leaf rust / leaf spot."
)

st.divider()

# ---------------------------------------------------------------- Bar chart
st.subheader("All 11 results, ranked")
st.caption("Test-set accuracy. Bars start at zero — axis isn't truncated.")

df = pd.DataFrame([
    {
        "label": f"{r['algorithm']}",
        "feature_set": FSET_LABEL[r["feature_set"]],
        "feature_set_key": r["feature_set"],
        "accuracy": r["accuracy"],
        "f1_macro": r["f1_macro"],
    }
    for r in results
])

chart = (
    alt.Chart(df)
    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
    .encode(
        x=alt.X("accuracy:Q", axis=alt.Axis(format="%", title="Test accuracy"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color(
            "feature_set:N",
            title="Feature set",
            scale=alt.Scale(domain=list(FSET_LABEL.values()), range=list(FSET_COLOR.values())),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Algorithm"),
            alt.Tooltip("feature_set:N", title="Feature set"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
            alt.Tooltip("f1_macro:Q", title="Macro F1", format=".3f"),
        ],
    )
    .properties(height=340)
)
text = chart.mark_text(align="left", dx=4, color="#5C6355").encode(text=alt.Text("accuracy:Q", format=".1%"))
st.altair_chart(chart + text, width="stretch")
st.caption("Majority-class baseline (always guess \"Leaf Rust\"): 44.7% — every model clears it comfortably.")

st.divider()

# ---------------------------------------------------------------- Winner detail
st.subheader(f"Winner in detail — {winner['algorithm']}")

col_cm, col_metrics = st.columns([1, 1], gap="large")

with col_cm:
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    cm = winner["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {short_names[c]}" for c in class_names],
        columns=[f"Pred: {short_names[c]}" for c in class_names],
    )
    st.dataframe(
        cm_df.style.background_gradient(cmap="Greens", vmin=0, vmax=cm_df.values.max()).format("{:d}"),
        width="stretch",
    )

with col_metrics:
    st.markdown("**Per-class metrics**")
    metrics_rows = []
    for cls in class_names:
        m = winner["per_class"][cls]
        metrics_rows.append({
            "Class": short_names[cls],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1-score"],
            "n": int(m["support"]),
        })
    metrics_df = pd.DataFrame(metrics_rows).set_index("Class")
    st.dataframe(
        metrics_df.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}", "n": "{:d}"}),
        width="stretch",
    )
    st.caption(
        "The soft spot: **Leaf spot** — the smallest class (41 test photos) — draws 6 false "
        "positives from Rust, pulling precision to 85%. Recall stays ≥95% on every class: the "
        "model rarely misses a disease outright, it occasionally mislabels which one."
    )

st.divider()

# ---------------------------------------------------------------- Full table
st.subheader("Full comparison table")
table_df = pd.DataFrame([
    {
        "Feature set": FSET_LABEL[r["feature_set"]],
        "Algorithm": r["algorithm"],
        "Accuracy": r["accuracy"],
        "Macro F1": r["f1_macro"],
    }
    for r in results
]).sort_values("Accuracy", ascending=False)
st.dataframe(
    table_df.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.3f}"})
    .background_gradient(subset=["Accuracy"], cmap="Greens"),
    width="stretch",
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------------- Notes
st.subheader("Reading the result")
st.markdown(
    """
- **Fine-tuning beats frozen embeddings, which beat hand-crafted features** — 96.7% vs 96.3% vs
  90.5% at each tier's best. The gap from hand-crafted → MobileNet embeddings (90.5% → 96.3%, same
  SVM) is the real story: the *representation* mattered most, not the classifier on top of it.
- **Once you have a decent embedding, the classifier barely matters** — SVM and Logistic Regression
  both land at 96.3% on MobileNet embeddings. kNN and Random Forest trail by 6–7 points on the same
  features, mostly losing ground on the minority "Leaf spot" class.
- A Kaggle discussion thread on this dataset flagged the Rust/Spot folders as possibly mislabeled.
  Visual inspection found the opposite of a swap: **Leaf Rust** shows the expected dense field of
  small pustules, **Leaf spot** shows fewer, larger discrete lesions — textbook presentations, just
  counter to what a first glance suggests.
- **Scope**: this dataset only covers 3 of the app's 25 tracked conditions (healthy / leaf rust /
  leaf spot). The other 22 — deficiencies, pests, less-common diseases — still run on the original
  synthetic-feature classifier until a labeled photo set exists for them too.
    """
)

st.caption(
    "Dataset: Nahiduzzaman et al., \"Explainable Deep Learning Model for Automatic Mulberry Leaf "
    "Disease Classification,\" *Frontiers in Plant Science* 14 (2023), CC0. Trained with "
    "scikit-learn + PyTorch/torchvision (CPU) — see `training/` for the reproducible script."
)
