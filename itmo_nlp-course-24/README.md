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

#### 1. Random (statistics-based)
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


#### 2. LSTM trainet on sequences of parts of speech
Location: `baselines/lstm_pos.ipynb`

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

Average Levenshtein distance: `7.433` (maximal is `19`)

**LSTM NN was badly trained/performed!**
Also it is possible that sequences of parts of speech are not enough for predicting sentence punctuation.

I'll try to fix the LSTM NN approach for the task in the second part of the project...

## Part 2

Here I decided to concentrate on predicting only _the intrinsic punctuation_: commas or spaces within the sentence!
Hence, in each target a finalizing mark (`.`, `!` or `?`) was replaced by the same token `F`
(replacement performed in `data-preparation/edit_target_intr_punct.ipynb`).

### New baselines

#### 1. Bigrams classification model
Location: `baselines/bigrams_classification.ipynb`

Model that classifies bigrams, where a class is a punctuation sign between two words in bigram.
Metrics are bad, but better tha metrics for incorrect LSTM-approach from previous part:
```
                  |comma (`,`)       |end of sent       |space (` `)       |
------------------|------------------|------------------|------------------|
Precision         |0.617602          |0.639752          |0.849541          |
Recall            |0.352839          |0.647217          |0.924959          |
F1 score          |0.449103          |0.643463          |0.885647          |
```



#### 2. X-Punctuator on lemmas sequences (THE BEST BASELINE)
Location: `baselines/lstm_xpunct_lemmas.ipynb`

The approach based on the X-Punctuation project from [github](https://github.com/kaituoxu/X-Punctuator/tree/master).
Sentences in form of lemmas sequencies were used as network input.
The obtained results are quite better (for commas prediction especially) than the previous ones:
```
                  |comma (`,`)       |end of sent       |space (` `)       |
------------------|------------------|------------------|------------------|
Precision         |0.621684          |1.000000          |0.893749          |
Recall            |0.467274          |1.000000          |0.940331          |
F1 score          |0.533531          |1.000000          |0.916449          |
```
Average Levenshtein distance: `1.689` (maximal is `13`)

### Models

#### 1. X-Punctuator based model with `navec` embedding
Location: `models/lstm_with_navec.ipynb`

For LSTM based network all **metrics were improved**
(except of small decreasing for end of sentence prediction)
in contrast to metrics of X-Punctuator baseline!
```
                  |comma (`,`)       |end of sent       |space (` `)       |
------------------|------------------|------------------|------------------|
Precision         |0.687129          |1.000000          |0.918454          |
Recall            |0.601647          |0.999102          |0.942514          |
F1 score          |0.641553          |0.999551          |0.930329          |
```
Average Levenshtein distance: `1.391` (maximal is `13`)

GRU-based network results:
```
                  |comma (`,`)       |end of sent       |space (` `)       |
------------------|------------------|------------------|------------------|
Precision         |0.723112          |1.000000          |0.909720          |
Recall            |0.547898          |1.000000          |0.955976          |
F1 score          |0.623428          |1.000000          |0.932275          |
```
Average Levenshtein distance: `1.368` (maximal is `10`)

A bit better than LSTM-based approach but results of both models depends on 
the randomized training process. 

I was unable to make the training process reproducible, 
but kept the model weights in `models/serialized` folder!

#### 2. From Hygging Face: [`markusiko/rubert-base-punctuation`](https://huggingface.co/markusiko/rubert-base-punctuation)
Location: `models/huggingface_model.ipynb`

The model is built upon the foundation of [ruBert-base](https://huggingface.co/ai-forever/ruBert-base) and has been fine-tuned to correctly place punctuation marks in Russian sentences (it predicts the mark after each word).

Some additional info about the model:

- **Fine-Tuning Source:** The model has undergone fine-tuning using a diverse dataset comprising over 20,000 paragraphs from Russian literary works. 
- **Supported Classes:** The model is designed to predict classes following specific punctuation marks: ? ! . , : ... and space (as class O).
- **Input Format:** To achieve optimal results, input text should be provided without punctuation marks. The model does not process changes in letter case.

Metrics _from the box_ (**all data**):
```
                |comma (`,`)     |excl. (`!`)     |point (`.`)     |question (`?`)  |space (` `)     |
----------------|----------------|----------------|----------------|----------------|----------------|
Precision       |0.913819        |0.405941        |0.763220        |0.567123        |0.975520        |
Recall          |0.774770        |0.232955        |0.960040        |0.824701        |0.983370        |
F1 score        |0.838569        |0.296029        |0.850390        |0.672078        |0.979430        |
```

Good results! Better metrics for commas and spaces classification.

Metrics on the selected **test** subset:
```
                |comma (`,`)     |excl. (`!`)     |point (`.`)     |question (`?`)  |space (` `)     |
----------------|----------------|----------------|----------------|----------------|----------------|
Precision       |0.925430        |0.428571        |0.784274        |0.623377        |0.975571        |
Recall          |0.782749        |0.200000        |0.967662        |0.827586        |0.986069        |
F1 score        |0.848131        |0.272727        |0.866370        |0.711111        |0.980792        |
```

#### 3. Fine-tuning of the model
Location: `models/finetuning.ipynb`\
[Code reference](https://github.com/Markusiko/RuPunctNet/blob/8bc765ddbd3f61822efc6ed6272fb5960dc8a37e/DL_experiments/bert-base.ipynb#L929)

Metrics on **all data**:
```
                |comma (`,`)     |excl. (`!`)     |point (`.`)     |question (`?`)  |space (` `)     |
----------------|----------------|----------------|----------------|----------------|----------------|
Precision       |0.971699        |0.802198        |0.973858        |0.860294        |0.992787        |
Recall          |0.965740        |0.414773        |0.989327        |0.932271        |0.994070        |
F1 score        |0.968710        |0.546816        |0.981532        |0.894837        |0.993428        |
```

Metrics on the same **test** subset:
```
                |comma (`,`)     |excl. (`!`)     |point (`.`)     |question (`?`)  |space (` `)     |
----------------|----------------|----------------|----------------|----------------|----------------|
Precision       |0.936912        |0.500000        |0.958587        |0.770492        |0.981751        |
Recall          |0.912668        |0.166667        |0.978856        |0.810345        |0.987088        |
F1 score        |0.924631        |0.250000        |0.968615        |0.789916        |0.984412        |
```

**Conclusion:** Metrics increased for all punctuation marks, except (maybe) for exclamation point!