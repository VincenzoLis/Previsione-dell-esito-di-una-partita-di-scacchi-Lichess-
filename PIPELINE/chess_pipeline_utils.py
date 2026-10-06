"""
Componenti riutilizzabili della pipeline di previsione pre-partita (Lichess).

Questo modulo NON fa parte dell'analisi del project work: contiene solo il
codice necessario per ricostruire ed eseguire la pipeline serializzata
(`pipeline_lr.joblib`). Deve restare importabile da un ambiente Python
qualsiasi, senza bisogno di riaprire il notebook, perche' joblib salva
solo un riferimento alla classe, non il suo codice: al momento del
caricamento (`joblib.load`) Python deve poter importare queste classi da
questo stesso file.

Il feature engineering replica ESATTAMENTE la logica dello Step 2 del
notebook (parsing di `increment_code`, feature derivate dai rating, stima
del tempo totale) e lo scaling della Sezione 0 dello Step 3 (StandardScaler
rifittato solo sul train interno). Nessun modello viene riaddestrato qui:
la pipeline riusa solo oggetti gia' addestrati e salvati in `artifacts/`.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Colonne grezze che devono essere presenti in input, disponibili prima
# dell'inizio della partita.
COLONNE_GREZZE_RICHIESTE = ["rated", "white_rating", "black_rating", "increment_code"]

# Ordine finale delle feature atteso dal modello (Regressione Logistica, modello
# scelto) - deve coincidere con l'ordine di X_train usato per l'addestramento nello Step 3.
ORDINE_FEATURE_FINALI = [
    "rated", "white_rating", "black_rating", "time_base", "increment",
    "rating_diff", "rating_mean", "rating_min", "rating_max", "time_total_estimate",
]

# Colonne numeriche scalate dallo StandardScaler salvato in artifacts/ (stesso
# ordine usato in fase di fit nello Step 3, Sezione 0).
COLONNE_NUMERICHE_DA_SCALARE = [
    "white_rating", "black_rating", "rating_diff", "rating_mean", "rating_min",
    "rating_max", "time_base", "increment", "time_total_estimate",
]


def _analizza_increment_code(codice):
    # Replica identica di parse_increment_code (Step 2 del notebook):
    # 'time_base+increment' -> (minuti, secondi); formati non validi -> (None, None).
    if isinstance(codice, str) and "+" in codice:
        parti = codice.split("+")
        try:
            time_base = int(parti[0])
            increment = int(parti[1])
            return pd.Series([time_base, increment])
        except ValueError:
            return pd.Series([None, None])
    return pd.Series([None, None])


class ChessPrePartitaFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Trasformatore stateless (nessun parametro appreso dai dati) che
    trasforma le colonne grezze pre-partita nelle feature usate dal modello.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        colonne_mancanti = set(COLONNE_GREZZE_RICHIESTE) - set(X.columns)
        if colonne_mancanti:
            raise ValueError(
                f"Colonne grezze mancanti in input: {sorted(colonne_mancanti)}. "
                f"Colonne richieste: {COLONNE_GREZZE_RICHIESTE}."
            )

        dati = X.copy()

        # 1. Parsing di 'increment_code' in 'time_base' (minuti) e 'increment' (secondi).
        dati[["time_base", "increment"]] = dati["increment_code"].apply(_analizza_increment_code)
        if dati[["time_base", "increment"]].isnull().any().any():
            raise ValueError(
                "Formato di 'increment_code' non riconosciuto per una o piu' righe "
                "(atteso 'time_base+increment', es. '10+5')."
            )

        # 2. 'rated' come intero 0/1.
        dati["rated"] = dati["rated"].astype(int)

        # 3. Feature derivate dai rating.
        dati["rating_diff"] = dati["white_rating"] - dati["black_rating"]
        dati["rating_mean"] = (dati["white_rating"] + dati["black_rating"]) / 2
        dati["rating_min"] = dati[["white_rating", "black_rating"]].min(axis=1)
        dati["rating_max"] = dati[["white_rating", "black_rating"]].max(axis=1)

        # 4. Stima del tempo totale disponibile (40 mosse per giocatore, come nello Step 2).
        dati["time_total_estimate"] = dati["time_base"] * 60 + dati["increment"] * 40

        return dati[ORDINE_FEATURE_FINALI]


class ChessScalerColonnare(BaseEstimator, TransformerMixin):
    """
    Applica un `StandardScaler` gia' addestrato solo alle colonne numeriche,
    lasciando invariata la colonna binaria 'rated'. Non rifitta mai lo
    scaler: usa solo `transform`, per non introdurre leakage.
    """

    def __init__(self, scaler, colonne_numeriche=COLONNE_NUMERICHE_DA_SCALARE):
        self.scaler = scaler
        self.colonne_numeriche = colonne_numeriche

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dati = X.copy()
        dati[self.colonne_numeriche] = self.scaler.transform(dati[self.colonne_numeriche])
        return dati[ORDINE_FEATURE_FINALI]
