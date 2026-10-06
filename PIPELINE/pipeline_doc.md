# Pipeline di inferenza — previsione esito partite di scacchi (Lichess)

Questo documento descrive gli artefatti prodotti nello **Step 5** del
project work e come riprendere il lavoro senza riaprire il notebook.
Il notebook contiene solo una cella di traccia; il codice che genera e usa
questi artefatti vive nei file elencati sotto, nella stessa cartella.

## File prodotti

| File | Cosa contiene |
|:--|:--|
| `chess_pipeline_utils.py` | Classi di feature engineering/scaling usate dalla pipeline. **Necessario** per poter ricaricare `pipeline_lr.joblib` con `joblib.load` (senza questo file l'unpickling fallisce). |
| `costruisci_pipeline_lr.py` | Script che costruisce la pipeline a partire dagli artefatti gia' addestrati in `artifacts/` e la salva su disco. Non riaddestra alcun modello. |
| `pipeline_lr.joblib` | Pipeline serializzata (preprocessing + Regressione Logistica, modello scelto, non calibrato), pronta all'uso. |
| `inference.py` | Funzione `predict_proba(input_df)` per fare previsioni su nuove partite. |
| `artifacts/` | Modelli gia' addestrati (logistica, GB conservativo, RF max_depth=10, MLP, modelli binari upset, scaler). Non modificarli. |
| `artifacts/v1.0/` | Prima versione "congelata" degli artefatti: `pipeline.joblib`, `metrics.json`, `thresholds.json`. |

## Come rigenerare la pipeline

```bash
python costruisci_pipeline_lr.py
```

Lo script legge `artifacts/standard_scaler.joblib` e
`artifacts/logistic_regression_multiclass.joblib` (gia'
addestrati nel notebook), li inserisce in uno sklearn `Pipeline` insieme al
feature engineering, e salva il risultato in `pipeline_lr.joblib`. Crea
anche (o aggiorna) la cartella `artifacts/v1.0/` con una copia congelata
della pipeline e dei file di riferimento.

**Nota sull'ambiente:** gli artefatti in `artifacts/` sono stati salvati con la
versione di scikit-learn usata dal kernel del notebook. Se l'ambiente Python
usato per eseguire questo script ha una versione diversa di scikit-learn,
il caricamento puo' fallire o generare avvisi di compatibilita' (es.
`InconsistentVersionWarning`): usare lo stesso ambiente/versione del notebook.

## Come usare la funzione di inferenza

```python
import pandas as pd
from inference import predict_proba

nuove_partite = pd.DataFrame({
    "rated": [True, False],
    "white_rating": [1500, 1200],
    "black_rating": [1450, 1600],
    "increment_code": ["10+5", "3+2"],
})

risultato = predict_proba(nuove_partite)
print(risultato)
#    black      draw     white decisione
# 0  ...        ...      ...   white
# 1  ...        ...      ...   black
```

**Input atteso:** un `pandas.DataFrame` con le colonne grezze disponibili
prima dell'inizio della partita — `rated` (bool/0-1), `white_rating` (int),
`black_rating` (int), `increment_code` (stringa `"time_base+increment"`,
es. `"10+5"`). Altre colonne eventualmente presenti vengono ignorate.

**Output:** un `pandas.DataFrame` con le stesse righe dell'input e le
colonne `black`, `draw`, `white` (probabilita', sommano a 1) e `decisione`
(la classe con probabilita' massima).

Per testare rapidamente la funzione, esegui `python inference.py`: viene
eseguito un test di esempio con due partite.

**Nota:** la funzione `predict_proba` accetta sia una singola partita sia
un batch di più partite: basta passare un DataFrame con più righe.
Restituisce un DataFrame con le stesse righe dell'input e le colonne delle
probabilità + decisione.

**Limite dichiarato:** la pipeline riproduce il modello Regressione Logistica
cosi' come descritto nel notebook (Step 3/4): le sue metriche di riferimento
provengono da una valutazione sul test temporale, consultato una sola volta
come stima indipendente nello Step 3, Sezione 6. Il modello non e' calibrato
esplicitamente: isotonic e Platt scaling sono stati testati (Step 3, Sezione 5)
ma nessuno dei due riduce l'ECE. Vedi i "Limiti metodologici" e
"Limiti residui" del notebook prima di un uso in produzione.

## Schema di versioning proposto

```
artifacts/
  v1.0/
    pipeline.joblib      # copia della pipeline di questa versione
    metrics.json         # metriche di riferimento sul test temporale
    thresholds.json       # regola decisionale e soglie di riferimento
  v1.1/
    pipeline.joblib
    metrics.json
    thresholds.json
  ...
```

Ogni cartella `vX.Y` è una fotografia immutabile di una versione della
pipeline:

- **`pipeline.joblib`**: la pipeline completa (feature engineering +
  scaling + modello) valida per quella versione.
- **`metrics.json`**: le metriche di riferimento (Log Loss, Brier Score,
  Accuracy, ECE medio, ECE della classe `draw`) calcolate al momento in cui
  la versione è stata creata, insieme a una nota sul dataset/periodo usato.
- **`thresholds.json`**: la regola decisionale (per il modello multiclass,
  `argmax`) ed eventuali soglie operative di riferimento (es. soglia upset
  del modello binario), cosi' come definite in quella versione.

**Convenzione dei numeri di versione:** incrementare la cifra dopo il punto
(`v1.0` → `v1.1`) per un ri-addestramento con gli stessi dati e la stessa
logica di feature engineering (es. nuovo tuning degli iperparametri).
Incrementare la cifra prima del punto (`v1.0` → `v2.0`) quando cambia la
logica di feature engineering, la scelta del modello, o la finestra
temporale di dati usata per l'addestramento.

**Come confrontare una nuova versione con la precedente:**

1. Calcolare le stesse metriche (`metrics.json`) della nuova versione
   sullo **stesso** set di valutazione usato per la versione precedente
   (o, meglio, su una finestra temporale indipendente non ancora usata da
   nessuna delle due versioni).
2. Confrontare Log Loss, Brier Score ed ECE medio/per classe: la nuova
   versione va promossa solo se migliora in modo credibile almeno una
   metrica probabilistica principale, senza peggiorare in modo rilevante
   le altre (vedi il confronto "champion/challenger" descritto nello
   Step 4 del notebook).
3. Se cambia la regola decisionale o una soglia in `thresholds.json`,
   documentare esplicitamente il motivo del cambiamento e da quale
   validazione proviene la nuova soglia.
4. Conservare sempre la versione precedente (`v1.0`, `v1.1`, ...) finché la
   nuova versione non è stata validata in produzione: permette un rollback
   immediato caricando di nuovo `pipeline.joblib` della versione precedente.
