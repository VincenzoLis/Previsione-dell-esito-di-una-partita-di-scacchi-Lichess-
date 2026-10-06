# ♟️ Previsione dell'esito di una partita di scacchi (Lichess)

<p align="left">
  <!-- Linguaggi e librerie -->
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=seaborn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Joblib-000000?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>

  <!-- Badge tematici -->
  <img src="https://img.shields.io/badge/Chess-000000?style=for-the-badge&logo=chess&logoColor=white"/>
  <img src="https://img.shields.io/badge/Lichess-000000?style=for-the-badge&logo=lichess&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-00C7B7?style=for-the-badge&logo=google&logoColor=white"/>
</p>

Sistema predittivo che stima, **prima dell’inizio della partita**, la probabilità di vittoria del Bianco, del Nero o di patta, utilizzando esclusivamente informazioni disponibili nella lobby: rating dei giocatori, tipo di partita (rated/casual) e formato di tempo.

Il progetto include:
- analisi esplorativa del dataset (20.000 partite Lichess)
- costruzione delle feature pre-partita
- confronto tra diversi modelli di classificazione
- scelta del modello migliore
- pipeline serializzata e pronta all’uso
- funzione di inferenza per nuove partite

---

## 📌 Obiettivo del progetto

1. Mostrare ai giocatori e agli spettatori una **probabilità pre-partita credibile**.  
2. Supportare il team matchmaking nell’identificare partite con **alto rischio di upset**.  
3. Fornire un modello **interpretabile, stabile e facilmente integrabile** in sistemi reali.

---

## 📊 Dataset

- **Fonte:** Chess Game Dataset (Lichess)  
- **File:** `games.csv`  
- **Dimensione:** ~20.000 partite  
- **Target:** `winner` (white / black / draw)  
- **Feature pre-partita utilizzate:**
  - `white_rating`
  - `black_rating`
  - `rated`
  - `increment_code`
  - differenza di rating
  - stime del tempo totale della partita

Split dei dati **temporale**, per simulare un uso realistico del modello in produzione.

---

## 🧠 Modelli testati

Sono stati confrontati diversi modelli:

- Regressione Logistica *(modello scelto)*
- Gradient Boosting
- Random Forest
- Albero decisionale
- Rete neurale MLP

La **Regressione Logistica** è risultata il miglior compromesso tra:
- qualità delle probabilità  
- calibrazione  
- interpretabilità  
- stabilità nel tempo  

---

## 🏆 Risultati principali

Valutazione sul **test temporale**:

| Metrica | Valore |
|--------|--------|
| Accuratezza | **60,6%** |
| Log Loss | **0,79** |
| Brier Score | **0,16** |
| Errore di calibrazione (ECE) | **0,018** |
| Errore calibrazione patta | **0,002** |

📌 *La differenza di rating è la feature più importante del modello.*  
📌 *La probabilità di patta è ben calibrata, ma la patta resta un esito raro.*

---

## 🏗️ Pipeline pronta all’uso

Il progetto include una pipeline **serializzabile e riutilizzabile**, composta da:

- Feature engineering + scaling (`StandardScaler`)
- Modello Regressione Logistica non calibrato
- Funzione di inferenza `predict_proba(input_df)`
- Versioning degli artefatti (`artifacts/v1.0/`)
- Documentazione tecnica (`pipeline_doc.md`)

### 📦 File principali

- `games.csv` — dataset delle partite  
- `Previsione_dell_esito_di_una_partita_di_scacchi_(Lichess).ipynb` — notebook completo  
- `pipeline_lr.joblib` — pipeline serializzata  
- `inference.py` — funzione di inferenza  
- `chess_pipeline_utils.py` — trasformatori e scaler  
- `costruisci_pipeline_lr.py` — script di costruzione pipeline  
- `pipeline_doc.md` — documentazione tecnica  
- `artifacts/` — modelli addestrati e versioni congelate  

### ▶️ Come usare la pipeline

```bash
python inference.py



📁 Struttura del repository
│
├── games.csv
├── report.md
├── pipeline_doc.md
├── Previsione_dell_esito_di_una_partita_di_scacchi_(Lichess).ipynb
├── inference.py
├── chess_pipeline_utils.py
├── costruisci_pipeline_lr.py
├── pipeline_lr.joblib
├── artifacts/
│   └── v1.0/
└── notebook.ipynb
