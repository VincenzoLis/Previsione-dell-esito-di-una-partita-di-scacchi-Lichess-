# Previsione dell'esito di una partita di scacchi (Lichess)
### Report di sintesi 

---

## 1. Scopo del progetto

Questo progetto costruisce un modello che stima, **prima che una partita di scacchi online inizi**, la probabilità che vinca il giocatore con i pezzi Bianchi, che vinca quello con i pezzi Neri, oppure che la partita finisca in patta. Il modello usa solo informazioni già disponibili nella lobby, prima della prima mossa: il rating dei due giocatori, se la partita è "rated" (con punteggio) o "casual", e il formato di tempo scelto.

Il modello serve a due scopi:

1. **Mostrare una probabilità pre-partita credibile** ai giocatori e agli spettatori, per rendere più trasparente e interessante l'attesa prima di una partita.
2. **Aiutare il team matchmaking** a capire quanto spesso il giocatore "sfavorito" (quello con rating più basso) riesce comunque a vincere — i cosiddetti **upset** — e a segnalare gli abbinamenti più a rischio di sorpresa.

Il progetto usa un dataset pubblico di circa 20.000 partite giocate su Lichess, con l'esito finale e le informazioni pre-partita (rating, tempo di gioco, apertura, tipo di partita).

---

## 2. Cosa è stato fatto

Il lavoro è organizzato in cinque fasi.

### 2.1 Esplorazione dei dati (Step 1)

Prima di costruire qualunque modello, i dati sono stati analizzati per capire cosa contengono e quali rischi comportano:

- **Distribuzione degli esiti**: il Bianco vince circa il 49,9% delle partite, il Nero circa il 45,4%, e solo il 4,7% finisce in patta. *(Grafico di riferimento: Step 1, "Distribuzione del target".)*
- **Rating e sorprese**: come atteso, più alta è la differenza di rating tra i due giocatori, più alta è la probabilità di vittoria del favorito. Tuttavia, in circa il 35% delle partite decisive con rating diversi vince comunque il giocatore sfavorito: le sorprese sono frequenti, non un'eccezione rara. *(Grafico di riferimento: Step 1, "Probabilità di vittoria/patta in base alla differenza di rating".)*
- **Aperture di gioco**: l'apertura scelta ha un legame con l'esito, ma debole e comunque non disponibile prima dell'inizio della partita, quindi non è stata usata come informazione pre-partita.
- **Partite rated vs casual e tempo di gioco**: emergono piccole differenze nei tassi di patta e di upset, ma non abbastanza forti da giustificare regole separate.
- **Qualità del dataset**: la maggior parte delle partite condivide giocatori con altre partite del dataset. Per questo motivo, invece di dividere i dati a caso, è stato usato uno **split temporale**: le partite più vecchie per addestrare i modelli, quelle più recenti per verificarli — una simulazione più realistica di come il modello verrebbe usato in produzione, su partite future.

### 2.2 Preparazione dei dati (Step 2)

A partire dai dati grezzi sono state costruite le informazioni (feature) usate dal modello: la differenza di rating tra i due giocatori, il rating medio, minimo e massimo, il formato di tempo e una stima del tempo totale della partita. Tutte le feature numeriche sono state riportate sulla stessa scala, un passaggio tecnico necessario per alcuni modelli, calcolato **solo sui dati di addestramento** per evitare che informazioni future "trapelassero" nella fase di training.

### 2.3 Costruzione e scelta del modello (Step 3)

Sono stati confrontati diversi tipi di modello: una regressione logistica (un modello semplice e interpretabile), alberi decisionali, foreste casuali, un modello di gradient boosting e una piccola rete neurale.

In sintesi, le differenze tra i modelli sono risultate contenute: la **Regressione Logistica** ha offerto il miglior equilibrio tra qualità delle probabilità e semplicità, ed è per questo il modello scelto; il **Gradient Boosting** ha ottenuto risultati quasi identici, ma la sua maggiore complessità non porta benefici sufficienti a giustificarla; la **Random Forest** è risultata leggermente peggiore su quasi tutte le misure; la **rete neurale (MLP)**, nonostante sia il modello più complesso, ha ottenuto le prestazioni peggiori tra i candidati principali; l'**albero decisionale singolo**, il più semplice, è stato il modello più debole.

Un punto centrale del metodo è come sono state prese le decisioni:

- I dati sono stati divisi in tre parti: **train** (per addestrare i modelli), **validation** (per confrontarli e scegliere il migliore, e per decidere se gestire lo squilibrio della classe "patta" o applicare una calibrazione) e **test** (mai toccato durante le decisioni, usato una sola volta alla fine come verifica indipendente).
- Per la classe "patta", che è rara, sono state provate tecniche per forzare il modello a predirla più spesso (pesi di classe, ricampionamento). Il risultato è stato che queste tecniche aumentano il numero di patte predette ma **peggiorano la qualità complessiva delle probabilità**: sono state quindi scartate.
- È stata inoltre testata una **calibrazione esplicita** delle probabilità (due tecniche standard, isotonic e Platt scaling) sul modello scelto: nessuna delle due migliora la qualità delle probabilità già prodotte dal modello, quindi non sono state applicate.
- Per ciascuna decisione (scelta del modello, scelta sulla patta, calibrazione, e più avanti il modello per gli upset) i dati di test sono stati consultati **una sola volta**, come verifica finale indipendente, mai per aggiustare le scelte.

Oltre al modello a tre esiti, è stata costruita anche una **versione semplificata a due esiti** (favorito vince / upset), pensata specificamente per il matchmaking, con una soglia di allerta calcolata tenendo conto che non segnalare un upset reale è considerato un errore più costoso di un falso allarme.

*(Grafici di riferimento: Step 3, matrici di confusione per ciascun modello; Step 3, reliability plot — i grafici che confrontano la probabilità dichiarata dal modello con la frequenza reale osservata.)*

### 2.4 Interpretazione e conclusioni (Step 4)

Il modello scelto è stato analizzato per capire quali informazioni pesano di più nelle sue previsioni, e sono state discusse le implicazioni pratiche per il prodotto, il matchmaking, il monitoraggio nel tempo e i limiti da tenere presenti. *(Grafico di riferimento: Step 4, "Importanza delle feature — Regressione Logistica".)*

### 2.5 Pipeline pronta all'uso (Step 5)

Infine, il modello scelto è stato impacchettato in una funzione pronta all'uso: basta fornire rating dei due giocatori, tipo di partita e formato di tempo, e la funzione restituisce le tre probabilità (Bianco, Nero, patta) e l'esito più probabile. Questo rende il modello riutilizzabile da altri sistemi senza dover riaprire il notebook di analisi.

---

## 3. Risultati principali

**Il modello scelto è la Regressione Logistica.** Tra tutti i modelli confrontati, è quello con il miglior equilibrio tra qualità delle probabilità (in particolare per la patta) e semplicità/interpretabilità. Il Gradient Boosting ottiene risultati quasi identici e resta un'alternativa valida, ma non offre vantaggi sufficienti da giustificare la maggiore complessità. Va detto che il criterio numerico che favorisce leggermente la Regressione Logistica (la qualità delle probabilità sulla patta) si basa su differenze piccole, misurate su appena 176 patte osservate nella fase di confronto: da solo è un segnale statisticamente debole. La scelta si appoggia perciò anche sulla sua semplicità e interpretabilità, un vantaggio pratico considerato importante quanto il margine numerico osservato.

Valutato **una sola volta** sui dati di test (mai usati prima in nessuna decisione), il modello scelto ottiene:

| Metrica | Valore | Cosa significa |
|:--|:--|:--|
| Accuratezza | 60,6% | Percentuale di partite in cui l'esito più probabile indicato dal modello è quello realmente accaduto |
| Log Loss | 0,79 | Misura di quanto le probabilità dichiarate sono vicine alla realtà (più basso è meglio) |
| Brier Score | 0,16 | Altra misura di qualità delle probabilità, più intuitiva (più basso è meglio) |
| Errore di calibrazione medio (ECE) | 0,018 | Quanto le probabilità dichiarate corrispondono, in media, alla frequenza reale (più basso è meglio) |
| Errore di calibrazione sulla patta | 0,002 | Le probabilità di patta dichiarate sono molto vicine alla frequenza reale osservata |

Per contestualizzare il Log Loss: una previsione **senza alcun modello** — basata solo sulle frequenze osservate delle tre classi, senza usare rating né altre informazioni della partita — otterrebbe su questi stessi dati un Log Loss di circa **0,85**. Il modello scelto (0,79) migliora quindi questo riferimento banale, pur lasciando margini di incertezza residua: il miglioramento è reale ma non enorme, coerente con uno sport dove anche il favorito netto può perdere.

**La differenza di rating tra i due giocatori è di gran lunga il segnale più importante** usato dal modello: pesa molto più di ogni altra informazione (formato di tempo, tipo di partita). Questo conferma in modo quantitativo quanto già visto nell'esplorazione iniziale dei dati. *(Grafico di riferimento: Step 4, importanza delle feature.)*

**La patta resta un esito difficile da prevedere come scelta singola.** Essendo rara (circa 1 partita su 20), il modello quasi non la sceglie mai come previsione principale. Questo non significa che la probabilità di patta sia sbagliata — anzi, è la probabilità meglio calibrata tra le tre — mostra soltanto che, quando si deve scegliere un solo esito, il modello preferisce quasi sempre una vittoria.

**La calibrazione è stata verificata e non necessita di correzioni aggiuntive.** Il modello scelto produce probabilità già ben allineate alla realtà; due tecniche di correzione standard sono state testate ma nessuna delle due porta benefici, quindi non sono state applicate.

**Il modello è disponibile anche come strumento pronto all'uso.** Oltre all'analisi descritta in questo report, il modello scelto è stato impacchettato in una funzione che, data una nuova partita (rating dei due giocatori, tipo di partita, formato di tempo), restituisce subito le tre probabilità, pronte per essere mostrate o utilizzate da altri sistemi. Questo pacchetto è **versionato**: ogni versione conserva una copia del modello insieme alle metriche e alle soglie di riferimento usate al momento del rilascio, per poter confrontare in futuro nuove versioni con quella attuale.

---

## 4. Cosa significa per prodotto e matchmaking

**Per la lobby e gli spettatori:**

- Mostrare sempre **tutte e tre le probabilità** (Bianco, Nero, patta), non solo l'esito più probabile: anche un favorito netto può perdere, e la patta ha un suo valore informativo anche quando è bassa.
- Presentare le probabilità come **stime**, con un messaggio semplice del tipo "stima basata soprattutto sui rating dei giocatori", per non dare l'impressione di un pronostico certo.
- Non nascondere né azzerare la probabilità di patta: è rara ma reale, e il modello la stima in modo coerente.

**Per il team matchmaking:**

- Il divario di rating resta il criterio più utile per costruire abbinamenti equilibrati, ma le sorprese (upset) sono frequenti — circa un terzo delle partite decisive con rating diversi — e vanno considerate come parte normale del gioco, non come anomalie.
- Il modello dedicato agli upset può essere usato per **segnalare** le partite più a rischio di sorpresa, così da dare priorità a eventuali controlli o analisi aggregate. Tra le partite segnalate come sospette, la quota di upset reali osservata nel test è circa il **46%**, contro un tasso di base di circa il **36%** tra tutte le partite decisive con rating diversi: un miglioramento di circa **10 punti percentuali (lift)** rispetto a una segnalazione casuale. Non deve però bloccare o modificare automaticamente un abbinamento: le soglie usate sono scelte sperimentali, non regole di prodotto definitive, e andranno concordate con il team in base al costo reale di un falso allarme rispetto a un upset non intercettato.
- Un upset non è, di per sé, prova di un errore di abbinamento o di un comportamento scorretto: significa solo che ha vinto il giocatore con rating più basso.

---

## 5. Limiti residui

Prima di considerare il modello pronto per un uso esteso in produzione, vanno tenuti presenti alcuni limiti:

- **La patta non viene mai scelta come previsione singola** dal modello, anche se la sua probabilità è ben calibrata. Se il prodotto ha bisogno di "segnalare" le partite a rischio patta in modo esplicito, serve una soglia dedicata, da definire insieme al team con i costi reali in mente — non è ancora stata implementata.
- **Il comportamento del gioco può cambiare nel tempo (drift).** Ad esempio, il tasso di upset osservato è già leggermente diverso tra il periodo di addestramento e quello di verifica (dal 34,3% al 36,5%). Non è detto che questa tendenza continui, ma va monitorata.
- **Le differenze tra i modelli migliori sono piccole e non è stata misurata la loro incertezza statistica** (per esempio con intervalli di confidenza): la scelta della Regressione Logistica è motivata e solida, ma non va considerata definitiva e andrebbe riconfermata con nuovi dati.
- **Tutte le verifiche si basano su un solo periodo storico**, diviso una sola volta in addestramento/validazione/test. Non è stato ancora verificato il comportamento del modello su più periodi diversi nel tempo.

---

## 6. Raccomandazioni

**Prima di un rilascio esteso:**

1. Verificare le prestazioni del modello (e delle soglie del modello upset) su una **finestra temporale nuova**, mai usata in nessuna fase di questo lavoro, per confermare che i risultati si mantengano.
2. Definire, insieme al team prodotto e matchmaking, i **costi reali** di un falso allarme rispetto a un upset mancato, per calibrare correttamente le soglie operative.
3. Decidere, con il team prodotto, come comunicare la probabilità di patta agli utenti finali (testo, percentuale, eventuale soglia dedicata).

**Per il monitoraggio continuo, una volta in produzione:**

1. Controllare **ogni settimana** la qualità delle probabilità (le stesse metriche riportate in questo report: Log Loss, Brier Score, errore di calibrazione), separatamente per Bianco, Nero e patta.
2. Seguire il **tasso di upset** osservato nel tempo e confrontarlo con le attese del modello: un aumento persistente è un segnale di possibile cambiamento nel comportamento dei giocatori o nella scala dei rating.
3. Programmare una **revisione periodica** (indicativamente trimestrale, o prima se le metriche peggiorano oltre soglie concordate) per decidere se ri-addestrare il modello con dati più recenti.
4. Gestire ogni aggiornamento della pipeline con un approccio **champion/challenger**: la versione attualmente in uso resta invariata finché una nuova versione candidata non viene confrontata, su dati indipendenti mai usati per costruirla, e non dimostra un miglioramento credibile senza peggiorare le altre metriche. Solo a quel punto la nuova versione sostituisce quella precedente.

---

*Questo report riassume in forma non tecnica il lavoro descritto nel notebook di analisi. Per i dettagli metodologici, le tabelle complete e tutti i grafici citati, fare riferimento alle sezioni corrispondenti del notebook (Step 1-5).*
