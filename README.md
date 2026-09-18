# 🎵 Music Learner Telemetry Diagnostic System
### End-to-End Supervised Machine Learning Pipeline & Interactive Deployment Interface

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science and machine learning pipeline designed to diagnose music student proficiency levels (`Beginner`, `Intermediate`, `Advanced`) using performance telemetry captured from individual practice sessions. 

The system features robust preprocessing, stratified cross-validation, algorithmic tournament benchmarking, coefficient explainability, and a live, reactive `ipywidgets` dashboard for real-time pedagogical assessment.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset & Telemetry Schema](#-dataset--telemetry-schema)
- [Pipeline Architecture (Stages 1–5)](#-pipeline-architecture-stages-15)
  - [Stage 1: Ingestion & Data Hygiene](#stage-1-ingestion--data-hygiene)
  - [Stage 2: Exploratory Data Analysis & Preprocessing](#stage-2-exploratory-data-analysis--preprocessing)
  - [Stage 3: Algorithmic Tournament & Selection](#stage-3-algorithmic-tournament--selection)
  - [Stage 4: Evaluation & Feature Attribution](#stage-4-evaluation--feature-attribution)
  - [Stage 5: Reactive UI Deployment](#stage-5-reactive-ui-deployment)
- [Key Findings & Pedagogical Insights](#-key-findings--pedagogical-insights)
- [Installation & Quickstart](#-installation--quickstart)
- [Appendix: Domain Adaptation & Cinematic Parallels](#-appendix-domain-adaptation--cinematic-parallels)
- [Author & License](#-author--license)

---

## 📖 Project Overview

Automating skill evaluation in music education often suffers from a reductionist bias—focusing entirely on mechanical metrics like note correctness while missing expressive interpretation. 

This project analyzes **2,000 discrete student practice sessions** across five acoustic and percussion instruments. By benchmarking linear models against ensemble decision trees and non-linear kernel transformations, this system confirms that while timing and pitch establish a necessary operational baseline, **expressiveness is the decisive statistical catalyst separating proficiency from true mastery**.

---

## 📂 Repository Structure

```
├── data/
│   └── music_learners_dataset.csv       # Raw session telemetry (2,000 records)
├── notebooks/
│   └── music_telemetry_pipeline.ipynb   # Complete 5-stage interactive notebook
├── assets/
│   ├── model_tournament.png             # Benchmark comparison visualization
│   ├── confusion_matrix.png             # Stage 4 test set diagnostic matrix
│   └── feature_coefficients.png         # Logistic regression weight breakdown
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Scaling and one-hot encoding pipelines
│   ├── train.py                         # Stratified CV and GridSearchCV routines
│   └── inference.py                     # Prediction engine & validation guardrails
├── requirements.txt                     # Project dependencies
├── LICENSE                              # MIT License
└── README.md                            # Project documentation
```

---

## 🎼 Dataset & Telemetry Schema

Each record captures a **single, discrete practice session** (not cumulative lifetime hours) with the following attributes:

| Feature Name | Data Type | Range / Domain | Description |
| :--- | :--- | :--- | :--- |
| `learner_id` | String / ID | Unique identifier | Identifier for individual student |
| `session_id` | String / ID | Unique identifier | Identifier for individual sit-down session |
| `instrument` | Categorical | `Piano`, `Guitar`, `Violin`, `Drums`, `Flute` | Instrument practiced |
| `age` | Integer | $8 - 70$ | Age of student |
| `practice_duration` | Float / Int | $20 - 120$ mins | Elapsed duration of that specific practice session |
| `tempo_accuracy` | Float | $0.0\% - 100.0\%$ | Telemetry-measured metronomic timing accuracy |
| `pitch_accuracy` | Float | $0.0\% - 100.0\%$ | Intonation / frequency alignment accuracy |
| `rhythm_accuracy` | Float | $0.0\% - 100.0\%$ | Subdivision and note-duration precision |
| `expressiveness_score` | Float | $1.0 - 10.0$ | Dynamic variation, phrasing nuance, and rubato |
| **`skill_level` (Target)** | Categorical | `Beginner`, `Intermediate`, `Advanced` | Operational competence label |

---

## ⚙️ Pipeline Architecture (Stages 1–5)

### Stage 1: Ingestion & Data Hygiene
* Loads raw session CSV telemetry into Pandas.
* Strips metadata keys (`learner_id`, `session_id`) to prevent target leakage.
* Audits schema types, confirms zero missing or null fields, and isolates numerical distributions.

### Stage 2: Exploratory Data Analysis & Preprocessing
* **Class Imbalance Audit:** Uncovers extreme distribution skew:
  * `Beginner`: 46.75% (935 sessions)
  * `Intermediate`: 46.85% (937 sessions)
  * `Advanced`: **6.40% (128 sessions)**
* **Train/Test Stratification:** 80/20 split ($1,600$ train / $400$ test) stratified along `skill_level` to preserve minority representation.
* **Leakage-Free Normalization:** Fits `StandardScaler` strictly on $X_{train}$ and transforms $X_{test}$.
* **Encoding:** One-hot encodes `instrument` attributes to feed linear decision surfaces.

### Stage 3: Algorithmic Tournament & Selection
Benchmarked three model families using **5-Fold Stratified Cross-Validation** followed by hyperparameter optimization via `GridSearchCV`:

```
========================================================================
Algorithm                 Baseline CV Acc     Tuned CV Acc    Optimal Hyperparameters
------------------------------------------------------------------------
Logistic Regression       98.06% (±0.0050)    99.38%          C=10.0, multinomial
Support Vector Machine    94.87% (±0.0132)    96.75%          C=10.0, gamma=0.01 (RBF)
Random Forest             90.19% (±0.0156)    90.25%          n_est=200, depth=None
========================================================================
```

> **Champion Selection:** **Multinomial Logistic Regression** won decisively. Musical competence progresses in a smooth, continuous diagonal trajectory; non-linear kernel transformations (SVM) and axis-aligned step functions (Random Forest) introduced unnecessary boundary complexity.

### Stage 4: Evaluation & Feature Attribution
Evaluated the Champion Model on the unseen 400-session test set:
* **Test Accuracy:** **99.75%** (1 single borderline Beginner/Intermediate error out of 400 instances).
* **Advanced Class Sensitivity:** 
  * **Precision:** 1.0000 (0 false positives)
  * **Recall:** 1.0000 (26 out of 26 Advanced cases detected)
  * **Macro F1-Score:** 0.9982
* **Coefficient Analysis:**
  * `expressiveness_score`: **$+13.3$ weight for Advanced**, **$-15.2$ for Beginner**.
  * `tempo`, `pitch`, `rhythm`: Stable baseline predictors ($+5.5$ to $+7.1$ for Advanced).
  * `practice_duration` & `age`: Coefficients hover near $\approx 0.0$, confirming session length does not dictate operational ability.

### Stage 5: Reactive UI Deployment
* Encapsulates data scaling, one-hot alignment, inference execution, and probability calibration into `diagnose_learner_session()`.
* Implements a full **reactive dashboard using `ipywidgets`**, rendering input sliders directly above an HTML diagnostic card that updates dynamically without notebook cell re-runs.

---

## 📊 Key Findings & Pedagogical Insights

```
                       [ ADVANCED MASTERY ]
                                ▲
                                │  +13.3 (Expressiveness Weight)
                       [ INTERMEDIATE TIER ]
                                ▲
                                │  +5.5 to +7.1 (Pitch, Rhythm, Tempo Floor)
                        [ BEGINNER TIER ]
```

1. **Mechanical Accuracy is the Floor, Not the Ceiling:**
   High tempo, pitch, and rhythm accuracy are required to reach Intermediate status, but they cannot promote a learner to Advanced status on their own.
2. **Artistry is the True Differentiator:**
   The model confirmed that `expressiveness_score` is the primary factor separating high-performing students from advanced artists.
3. **Session Length Does Not Equal Competence:**
   Individual session duration carries virtually zero predictive power. Deliberate, expressive practice matters far more than seated time.

---

## 💻 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/music-telemetry-diagnostics.git](https://github.com/your-username/music-telemetry-diagnostics.git)
cd music-telemetry-diagnostics
```

### 2. Configure Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch Notebook or Run Standalone Inference
Launch Jupyter / Google Colab:
```bash
jupyter notebook notebooks/music_telemetry_pipeline.ipynb
```

Or execute inline inference in Python:
```python
from src.inference import diagnose_learner_session

report = diagnose_learner_session(
    age=22,
    instrument="Violin",
    practice_duration=60,
    tempo_accuracy=94.5,
    pitch_accuracy=92.0,
    rhythm_accuracy=93.5,
    expressiveness_score=8.8
)
print(report["predicted_tier"], report["confidence"])
```

---

## 🎭 Appendix: Domain Adaptation & Cinematic Parallels

### Cinematic Reflections on Measurement vs. Artistry
* **The J. Evans Pritchard Fallacy (*Dead Poets Society*):** Plotting art on an X/Y grid of "importance vs. perfection" fails because technical perfection alone cannot define aesthetic impact.
* **The Benchmark of Genius (*I, Robot*):** When Spooner asks if a robot can turn a canvas into a masterpiece, Sonny responds: *"Can you?"* Statistically, artistic mastery is rare; only 6.4% of sessions in this dataset attained Advanced expression.
* **The Feedback Loop Breakdown (*RoboCop* 2014):** A cybernetic arm plays the guitar flawlessly until human emotion causes unexpected motor variation, causing the system to crash. Deterministic systems often struggle when confronted with organic human feeling.

### Domain Adaptation: Voice as an Instrument
Deploying this architecture to **classical vocal telemetry** alters core feature dynamics:
* **Vibrato Compensation:** Voice requires signal processing to isolate natural pitch oscillation (5–7 Hz vibrato) from genuine pitch drift.
* **Fatigue Inversion:** Extended single-session duration on vocal cords creates physiological fatigue, transforming duration from a neutral weight into a potential negative penalty.
* **Acoustic Formants:** Demands tracking spectral acoustic cues like the **Singer’s Formant ($2.5–3.2\text{ kHz}$)** to gauge vocal ring and projection.

---

## 👤 Author & License

* **Developer:** Cesar Juarez
* **Framework:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, IPyWidgets
* **License:** This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
