"""
Script per costruire e salvare su disco la pipeline serializzabile del
modello di previsione pre-partita (Lichess).

Uso:
    python costruisci_pipeline_lr.py

Cosa fa:
    1. Carica lo StandardScaler e la Regressione Logistica (modello scelto,
       Step 3 Sezione 6) gia' addestrati e salvati in artifacts/ (NON
       riaddestra nulla).
    2. Costruisce uno sklearn Pipeline che concatena:
       feature engineering -> scaling -> modello.
    3. Salva la pipeline in questa stessa cartella come 'pipeline_lr.joblib'.
    4. Crea una prima versione degli artefatti in 'artifacts/v1.0/'
       (pipeline, metriche e soglie di riferimento), secondo lo schema di
       versioning descritto in pipeline_doc.md.

Le metriche in metrics.json sono quelle gia' calcolate e discusse nello
Step 3 (Sezione 6, valutazione finale sul test) del notebook: sono
riportate qui come riferimento congelato per questa versione, non
ricalcolate ad ogni esecuzione.
"""

import json
import shutil
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from chess_pipeline_utils import (
    ChessPrePartitaFeatureEngineer,
    ChessScalerColonnare,
    COLONNE_NUMERICHE_DA_SCALARE,
)

CARTELLA_CORRENTE = Path(__file__).resolve().parent
CARTELLA_ARTIFACTS = CARTELLA_CORRENTE / "artifacts"
PERCORSO_SCALER = CARTELLA_ARTIFACTS / "standard_scaler.joblib"
PERCORSO_MODELLO = CARTELLA_ARTIFACTS / "logistic_regression_multiclass.joblib"
PERCORSO_PIPELINE_FINALE = CARTELLA_CORRENTE / "pipeline_lr.joblib"

VERSIONE_CORRENTE = "v1.0"

# Metriche di riferimento della Regressione Logistica sul test temporale,
# usato una sola volta come valutazione finale indipendente (Step 3, Sezione 6).
METRICHE_RIFERIMENTO = {
    "modello": "logistic_regression_multiclass",
    "log_loss": 0.7876,
    "brier_score_medio": 0.1622,
    "accuracy": 0.6064,
    "ece_medio": 0.0184,
    "ece_draw": 0.0018,
    "nota": (
        "Metriche calcolate sul test temporale, consultato una sola volta come "
        "valutazione finale indipendente del modello scelto sulla validation "
        "(Step 3, Sezione 6). Isotonic e Platt scaling sono stati testati come "
        "calibrazione esplicita (Step 3, Sezione 5) ma non applicati: nessuno dei "
        "due riduce l'ECE rispetto al modello non calibrato."
    ),
}

# Regola decisionale della pipeline multiclass: nessuna soglia, si sceglie la
# classe con probabilita' massima. La soglia 0.38 del modello binario di upset
# e' riportata solo come riferimento: quel modello e' un artefatto separato
# (logistic_regression_upset_binary.joblib), non incluso in pipeline_lr.joblib.
SOGLIE_RIFERIMENTO = {
    "decisione_multiclasse": "argmax delle 3 probabilita (white / black / draw)",
    "soglia_upset_logistic_regression_binario": 0.38,
    "soglia_upset_gradient_boosting_binario": 0.33,
    "nota": (
        "Le soglie upset provengono dal modello binario dello Step 3 "
        "(rilevazione upset) e non sono usate da pipeline_lr.joblib."
    ),
}


def costruisci_pipeline():
    print("--- Costruzione della pipeline serializzabile (feature engineering + scaling + modello) ---")

    scaler_originale = joblib.load(PERCORSO_SCALER)
    modello_originale = joblib.load(PERCORSO_MODELLO)

    pipeline_lr = Pipeline(steps=[
        ("feature_engineering", ChessPrePartitaFeatureEngineer()),
        ("scaling", ChessScalerColonnare(scaler_originale, COLONNE_NUMERICHE_DA_SCALARE)),
        ("modello", modello_originale),
    ])

    joblib.dump(pipeline_lr, PERCORSO_PIPELINE_FINALE)
    print(f"Pipeline salvata in: {PERCORSO_PIPELINE_FINALE}")
    return pipeline_lr


def crea_versione_artifacts(versione=VERSIONE_CORRENTE):
    cartella_versione = CARTELLA_ARTIFACTS / versione
    cartella_versione.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(PERCORSO_PIPELINE_FINALE, cartella_versione / "pipeline.joblib")

    with open(cartella_versione / "metrics.json", "w", encoding="utf-8") as file_metriche:
        json.dump(METRICHE_RIFERIMENTO, file_metriche, indent=2, ensure_ascii=False)

    with open(cartella_versione / "thresholds.json", "w", encoding="utf-8") as file_soglie:
        json.dump(SOGLIE_RIFERIMENTO, file_soglie, indent=2, ensure_ascii=False)

    print(f"Versione '{versione}' degli artefatti creata in: {cartella_versione}")


if __name__ == "__main__":
    costruisci_pipeline()
    crea_versione_artifacts()
