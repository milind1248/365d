# 365Dfarms Mulberry AI — Streamlit Web Edition

A browser-based port of the `365D-Farm` Flutter app: AI-assisted mulberry
leaf disease/pest/deficiency scanning, farm plot & records management,
weather-based spray guidance and expert help — built to run on
[Streamlit Community Cloud](https://streamlit.io/cloud).

## What's inside

| Area | How it works |
|---|---|
| **Database** | SQLite (`core/db.py`), schema ported 1:1 from the Flutter app's `database_helper.dart` (`user_profile`, `farm_plot`, `scan_record`, `spray_log`, `fertilizer_log`, `soil_test`, `crop_task`, `production_record`, `expert_case`, `notification_item`). Auto-created on first run. |
| **Auth** | `core/auth.py` — local register/login/guest, salted PBKDF2 password hashes, session via `st.session_state`. |
| **AI / ML disease detection** | Two models, blended in `ai/classifier.py`. **Real-photo model**: `ai/mobilenet_classifier.py`, a MobileNetV3-Small fine-tuned on 1,091 real Kaggle leaf photos — 96.7% test accuracy on healthy/rust/spot. **Synthetic model**: `ai/features.py` extracts real color/texture features from the photo, `ai/train_model.py` trains a scikit-learn `RandomForestClassifier` on all 25 classes for the conditions with no real photo dataset yet. See "About the AI model" below. |
| **Severity estimator** | `ai/severity.py` — pixel-level heuristic ported from the Flutter app's `SeverityEstimator`, estimates % affected leaf area. |
| **Disease risk engine** | `ai/disease_risk.py` — rule-based mildew/rust/fungal/pest risk from weather, ported from `disease_risk_engine.dart`. |
| **Weather** | `ai/weather.py` — free, keyless [Open-Meteo](https://open-meteo.com) forecast API, with a deterministic mock fallback if offline. |
| **Chatbot** | `ai/chatbot.py` — [Groq](https://groq.com) (Llama 3.3 70B) as the primary answer engine, grounded with matching knowledge-base entries so treatment/dosage advice stays consistent; falls back automatically to the original offline keyword-match engine (ported from `chatbot_engine.dart`) if no API key is configured or the call fails. |
| **Knowledge base** | `data/diseases_knowledge_base.json`, `data/labels.txt` — copied from the Flutter app's `assets/`, so both apps share the same 25 disease/pest/deficiency definitions. |
| **Plot location picker** | `core/geocoding.py` — resolves an Indian PIN code or city name to lat/lon via India Post + Open-Meteo's free geocoding API (with a state-centroid fallback for renamed-city mismatches), plus a "use my GPS" button (`streamlit-geolocation`) — both feed the Latitude/Longitude fields in **My Farm → Add a new plot**, which drive the weather/disease-risk forecasts. |
| **User management (admin)** | `app_pages/admin.py` — **Account → User Management**, gated by a hardcoded password (`365D` — see warning below). Lists every account (name, contact, guest/registered, created date, last login), with revoke/restore access and permanent delete (cascades the user's plots/scans/logs/records). Revoking mid-session force-logs-out that user on their next click. |
| **Persistent login** | `core/session_cookie.py` + `core/auth.py` — a "remember me" browser cookie (30 days, `REMEMBER_ME_DAYS`) backed by a hashed token in the `user_session` table, so a page reload or new tab doesn't log the farmer out. |
| **Keep-alive ping** | `.github/workflows/ping-streamlit.yml` — a scheduled GitHub Actions job (every 10 minutes) that pings the deployed app so Streamlit Community Cloud doesn't put it to sleep from inactivity. **Edit the placeholder URL in that file once the app is deployed** (see comment inside). |
| **Silkworm AI scan** | `app_pages/scan_silkworm.py` + `ai/silkworm_classifier.py` — a second, independent MobileNetV3-Small model fine-tuned on real silkworm larva photos, for healthy / Grasserie / generic-infected detection. See "About the silkworm AI model" below. |
| **Silkworm rearing tracking** | `app_pages/silkworm_tracking.py` — log rearing batches (instar stage, mortality %, last-fed time); feeds the Alert Engine's feeding/instar/mortality reminders. |
| **Alert Engine** | `ai/alert_engine.py` — categorizes and prioritizes every notification (disease/climate/feeding/instar/mortality/general × critical/high/medium/low), with dedup so the same condition doesn't re-alert every rerun. **Account → Notifications** got a full rework: filter by category/status, mark alerts completed/skipped with an optional outcome note. The Dashboard shows a live open-alerts summary by priority. |

## ⚠️ Admin access is hardcoded — fix before any real deployment

`app_pages/admin.py`'s password (`365D`) is a plain string in the source
code, not a secret, not hashed, and not tied to any real account — anyone
who reads the repo (it's public once pushed to GitHub) knows it. This is
fine for local development/demos only. Before deploying anywhere a stranger
could reach the URL:
- Move the password into `.streamlit/secrets.toml` (same pattern as the
  Groq key — see "Chatbot setup" below) at minimum, or
- Better: replace the whole gate with a real admin account (a `user_profile`
  row with an `is_admin` flag, checked via the existing hashed-password
  `core/auth.py` flow) so it's not a single shared secret at all.

## About the AI model

**For healthy leaves, leaf rust, and leaf spot** — the 3 conditions with a real
labeled photo dataset ([Kaggle, CC0](https://www.kaggle.com/datasets/nahiduzzaman13/mulberry-leaf-dataset)) —
diagnosis runs on a **MobileNetV3-Small fine-tuned end-to-end on 1,091 real
photos**, measured at **96.7% test accuracy** (0.953 macro-F1) on a held-out
273-photo split. It won a head-to-head comparison against 10 other
combinations: hand-crafted color/texture features and frozen MobileNet
embeddings, each run through Random Forest, SVM, Logistic Regression,
Gradient Boosting, and k-NN. Full results, per-model confusion matrices, and
the training/comparison script are in `training/` (see `training/README.md`
to reproduce).

**For the other 22 conditions** (pests, nutrient deficiencies, stress) —
where no labeled photo dataset exists — `ai/train_model.py` trains a
scikit-learn `RandomForestClassifier` on feature distributions derived from
the documented symptom descriptions (e.g. "white powdery coating" → high
white/gray pixel ratio), and `ai/classifier.py` runs `predict_proba()` on
real pixel features extracted from the uploaded photo. Held-out accuracy on
this synthetic validation data: ~80%.

`ai/classifier.py` blends the two: it always evaluates the 25-class synthetic
model first, and swaps in the MobileNet result whenever the top guess lands
on one of its 3 covered classes. Either way, **this is advisory
pattern-matching, not clinical diagnosis.** To extend real-photo coverage to
more conditions, add labeled images for them and retrain following
`training/README.md`.

## About the silkworm AI model

Silkworm scanning (**Farm → Scan Silkworm**) uses a **separate** model trained
on **4,862 real photos** from the Kaggle
[`tatinanikhitha/silkworm`](https://www.kaggle.com/datasets/tatinanikhitha/silkworm)
dataset (Andhra Pradesh sericulture field data): 2,379 healthy, 2,286 generic
"infected", and 197 confirmed **Grasserie** (a specific, named viral disease).
It was compared against 10 other model/feature combinations (hand-crafted
color/texture features, and a fully fine-tuned MobileNetV3-Small, each also
run through 5 classical algorithms) — see `training/train_silkworm_compare.py`
and `data/silkworm_training_results.json` for the full comparison and
confusion matrices. Unlike the mulberry model, the winner here was **not**
the fine-tuned network: **frozen MobileNetV3-Small (ImageNet) embeddings fed
into an SVM (RBF)** scored highest at **97.9% test accuracy** (0.908
macro-F1) on a held-out 1,215-photo split — narrowly ahead of the fine-tuned
model's 97.8%, and picked by the same strict test-accuracy tie-break rule
used for the mulberry model, for consistency.

**Honesty note:** the source dataset only names one disease (Grasserie)
individually — its other diseased photos are labeled generically as
"Infected," not broken into Flacherie, Muscardine, Pébrine, etc. A result of
"signs of illness (unconfirmed cause)" means the model is confident
something is wrong, not which specific disease — `data/silkworm_knowledge_base.json`'s
`infected_silkworm` entry says this explicitly and steers the farmer to
**Submit to Expert** rather than guessing. Unlike the mulberry model, there's
no synthetic-fallback classifier here: silkworm body-color heuristics aren't
a validated signal the way leaf-color heuristics are, so below-confidence
results show "inconclusive" instead of inventing a diagnosis.

## Chatbot setup (Groq)

The chatbot works out of the box with the offline knowledge-base engine even
with no key configured. To enable the Groq-powered answers:

1. Get a free API key from [console.groq.com/keys](https://console.groq.com/keys).
2. Create `.streamlit/secrets.toml` (gitignored — never commit this file):
   ```toml
   [llm]
   groq_api_key = "gsk_..."
   ```
   Alternatively, set the `GROQ_API_KEY` environment variable — `ai/chatbot.py`
   checks `st.secrets` first, then falls back to the env var.

## Running locally

```bash
pip install -r requirements.txt
python -m ai.train_model   # optional - classifier.py auto-trains on first use if missing
streamlit run streamlit_app.py
```

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (root = this folder, or set it as
   the app's "Main file path" base).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `streamlit_app.py` in that repo.
3. The weather API is keyless and the disease-detection models are committed,
   so the app runs with **no secrets configured at all** — the chatbot just
   falls back to its offline engine. To enable Groq answers, open the app's
   **Settings → Secrets** in Streamlit Community Cloud and paste the same
   `[llm]` block shown above.

### Persistent storage caveat

Streamlit Community Cloud's filesystem is **ephemeral** — it resets whenever
the app redeploys or wakes from sleep. The bundled SQLite database
(`data/mulberry_ai.db`) and uploaded scan photos (`data/uploads/`) are
therefore fine for demos but **not durable** across redeploys. For a
production deployment, swap `core/db.py`'s `get_connection()` for a hosted
database (e.g. Supabase/Postgres via `st.connection`, or Turso/libSQL) — the
rest of the app talks to `core/db.py`'s functions, not to SQLite directly, so
only that one module needs to change.

## Project layout

```
streamlit_app.py         # entry point: auth gate + navigation
app_pages/                # one file per page (dashboard, scan, farm, logs, ...)
  scan_silkworm.py         # silkworm disease AI scan page
  silkworm_tracking.py     # silkworm rearing batch tracking (instar/feeding/mortality)
core/
  db.py                  # SQLite schema + CRUD helpers
  auth.py                # register/login/guest session, access-revocation check
  session_cookie.py        # "remember me" persistent login cookie
  helpers.py              # shared UI helpers (badges, plot pickers, notifications)
  geocoding.py             # PIN code / city -> lat/lon (India Post + Open-Meteo geocoding)
ai/
  features.py             # real pixel feature extraction
  severity.py              # affected-area heuristic
  classifier.py            # blends the two mulberry models + inference
  mobilenet_classifier.py  # real-photo-trained MobileNetV3-Small (healthy/rust/spot)
  train_model.py           # trains & saves the synthetic-feature RandomForest (25 classes)
  silkworm_classifier.py   # real-photo-trained MobileNet-embed + SVM for silkworm disease
  alert_engine.py           # categorized/prioritized notification generation
  disease_risk.py          # weather -> risk rules
  weather.py                # Open-Meteo forecast client
  chatbot.py                 # Groq-powered chatbot, offline knowledge-base fallback
training/                 # dataset download/extraction + model comparison scripts, see training/README.md
data/                     # knowledge base, labels, trained models, SQLite DB (gitignored), uploads (gitignored)
assets/images/logo.png    # brand logo
.github/workflows/ping-streamlit.yml  # scheduled keep-alive ping (edit URL after deploying)
```
