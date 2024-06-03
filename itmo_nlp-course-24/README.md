## ITMO course "Natural language analysis using machine learning methods" (apr 2024)

**Project topic (en):** Punctuation Prediction for Sentences in Russian \
**Тема проекта (rus):** Расстановка знаков препинания в предложениях **на русском языке**

Data was taken [from kaggle](https://www.kaggle.com/datasets/d0rj3228/russian-literature?resource=download)

Python version: `3.8.10`\
Requirements: `requirements.txt`

## Part 1

### Data

1. [Alexander Pushkin's](https://en.wikipedia.org/wiki/Alexander_Pushkin) prose (_raw books_ are located in `data/raw/pushkin`)
   - **Дубровский** [en. Dubrovsky] (1831): `dubrovsky.txt`
   - **Повести покойного Ивана Петровича Белкина** [en. The Tales of the Late Ivan Petrovich Belkin] (1831): 
   `povesti_belkina.txt`
   - **История Пугачева** [en. A History of Pugachev] (1834): `pugachevs_story.txt`
   - **Капитанская дочка** [en. The Captain's Daughter] (1836): `kapitanskaya_dochka.txt`

### Data preparation
Data preparation for further punctuation prediction was performed. 

1. all books by **Alexander Pushkin** were prepared by `data-preparation/pushkin_common.ipynb`;
2. [`razdel`](https://github.com/natasha/razdel) by Natasha Project was used to split texts by sentences;
3. sentences were lowercased and cleaned from extra symbols, numbers were saved in texts;
4. sentences were **tokenized**
   - punctuation within a sentence was limited only by `commas` (all other marks were substituted by `,`);
   - punctuation in the end of sentences was saved: `point`, `exclamation` or `question`;
5. `DataFrames` of obtained and tokenized sentences from each book were saved in `data/prepared/pushkin` directory in `.csv` format;
6. data from all Pushkin's book was accumulated and selected during small EDA in `eda/eda_pushkin.ipynb`
   - final set of selected sentences was saved in `data/prepared/01_punct_pushkin.csv`

Example of the sentence preparation and tokenization:
```bash
Sentence: '«Ну, Савельич, — сказал я ему, — отдай же мне теперь половину; а остальное возьми себе.'
Cleaning: 'ну, савельич, сказал я ему, отдай же мне теперь половину, а остальное возьми себе.'

Input   : 'ну савельич сказал я ему отдай же мне теперь половину а остальное возьми себе'
Target  : 'C C S S C S S S S C S S S P'
```

### Baselines

#### Random (statistics-based)
Location: `baselines/random.ipynb`

Predicting of punctuation marks with probabilities obtained from marks occurancy statistics in train dataset. 

_**Comment**_: Generation of _intricic_ (` ` or `,`) and _finalyzing_ (`.`, `!` or `?`) punctuation marks was performed separately!

Metrics (classification)
```
                  |comma (`,`)       |exclamation (`!`) |period (`.`)      |question (`?`)    |space (` `)       |
------------------|------------------|------------------|------------------|------------------|------------------|
Precision         |0.177802          |0.050618          |0.899730          |0.064874          |0.827461          |
Recall            |0.178674          |0.060000          |0.900998          |0.055556          |0.826560          |
F1 score          |0.178208          |0.054624          |0.900349          |0.059683          |0.827004          |
```

Average Levenshtein distance between predictions and targets: `3.671` (while average sentence len in test dataset was `12.94`)


#### LSTM trainet on sequences of parts of speech
Location: `baselines/lstm_speech_parts.ipynb`

Example of the sentence preparation and tokenization:
```bash
Input      : 'адъюнкт иноходцев бывший тут же успел убежать'
Input (pos): 'ADJ VERB ADJ ADV PART VERB VERB'

Target  : 'S C S S C S P'
```

Metrics (classification)
```
                  |comma (`,`)       |exclamation (`!`) |period (`.`)      |question (`?`)    |space (` `)       |
------------------|------------------|------------------|------------------|------------------|------------------|
Precision         |0.190476          |nan               |0.110393          |nan               |0.923811          |
Recall            |0.067620          |0.000000          |0.886228          |0.000000          |0.466527          |
F1 score          |0.099808          |0.000000          |0.196330          |0.000000          |0.619969          |
```

Average Levenshtein distance: `7.433`

**LSTM NN was badly trained/performed!**
Also it is possible that sequences of parts of speech are not enough for predicting sentence punctuation.

I'll try to fix the LSTM NN approach for the task in the second part of the project...

## Part 2

