"""
Funzione di inferenza per il modello di previsione esito partite di scacchi
(Lichess), pensata per essere usata senza riaprire il notebook.

Uso tipico:

    import pandas as pd
    from inference import predict_proba

    nuove_partite = pd.DataFrame({
        "rated": [True],
        "white_rating": [1500],
        "black_rating": [1450],
        "increment_code": ["10+5"],
    })
    risultato = predict_proba(nuove_partite)
    print(risultato)

Input atteso (predict_proba):
    Un pandas.DataFrame con le colonne grezze disponibili PRIMA dell'inizio
    della partita:
        - rated (bool o 0/1): partita valutata o casual.
        - white_rating (int): rating Elo del Bianco.
        - black_rating (int): rating Elo del Nero.
        - increment_code (str): formato 'time_base+increment', es. '10+5'.
    Puo' contenere anche altre colonne (es. id partita): vengono ignorate.

Output (predict_proba):
    Un pandas.DataFrame con lo stesso indice dell'input e le colonne:
        - 'black', 'draw', 'white': probabilita' dei tre esiti (sommano a 1).
        - 'decisione': la classe con probabilita' massima tra le tre.
"""

from pathlib import Path

import joblib
import pandas as pd

# Import necessario solo per rendere disponibili le classi custom usate da
# pipeline_lr.joblib al momento del caricamento (joblib deve poterle importare).
from chess_pipeline_utils import ChessPrePartitaFeatureEngineer, ChessScalerColonnare  # noqa: F401

_CARTELLA_CORRENTE = Path(__file__).resolve().parent
_PERCORSO_PIPELINE = _CARTELLA_CORRENTE / "pipeline_lr.joblib"

_pipeline = None  # caricata pigramente al primo utilizzo, poi tenuta in cache


def _carica_pipeline():
    global _pipeline
    if _pipeline is None:
        if not _PERCORSO_PIPELINE.exists():
            raise FileNotFoundError(
                f"Pipeline non trovata in {_PERCORSO_PIPELINE}. "
                "Eseguire prima 'python costruisci_pipeline_lr.py'."
            )
        _pipeline = joblib.load(_PERCORSO_PIPELINE)
    return _pipeline


def predict_proba(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le probabilita' pre-partita (white / black / draw) e la
    decisione (classe piu' probabile) per un DataFrame di nuove partite.
    Vedi la documentazione del modulo per le colonne di input attese.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df deve essere un pandas.DataFrame.")

    pipeline = _carica_pipeline()
    probabilita = pipeline.predict_proba(input_df)
    classi = pipeline.named_steps["modello"].classes_

    risultato = pd.DataFrame(probabilita, columns=classi, index=input_df.index)
    risultato["decisione"] = risultato[classi].idxmax(axis=1)
    return risultato


if __name__ == "__main__":
    # Test rapido manuale: due partite di esempio con le sole colonne grezze pre-partita.
    partite_di_esempio = pd.DataFrame({
        "rated": [True, False],
        "white_rating": [1500, 1200],
        "black_rating": [1450, 1600],
        "increment_code": ["10+5", "3+2"],
    })
    print(predict_proba(partite_di_esempio))
