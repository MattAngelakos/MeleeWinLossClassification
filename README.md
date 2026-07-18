# Super Smash Bros. Melee Win/Loss Classification

Binary classification of Melee tournament matches: given only **pre-game** attributes
(who is playing, which characters, which stage, where in the bracket), predict whether
player 1 won or lost.

CS 513-B final project — Matthew Angelakos, Tri Luu (Group 22).

Data: [Super Smash Bros. database](https://www.kaggle.com/datasets/thomasdubail/super-smash-bros-database/)
(Kaggle), tournament sets from 2018–2023. After cleaning and expanding each set into its
individual games, 1,083,300 rows remain.

## Results

**Random Forest is the best model at 70% accuracy.** It is also the only model that
clearly beats the baseline — see the caveat below.

Evaluated on a 216,661-row held-out test set (20% split, `random_state=42`):

| Model | Accuracy | Win F1 | Loss F1 |
|---|---|---|---|
| Majority-class baseline (always predict "win") | 63.1% | 0.77 | 0.00 |
| ANN (`MLPClassifier`, defaults) | 64% | 0.76 | 0.27 |
| Decision Tree (`DecisionTreeClassifier`, defaults) | 65.0% | 0.72 | 0.54 |
| AdaBoost (depth 11, 110 estimators) | 67% | — | — |
| **Random Forest (100 trees)** | **70%** | **0.77** | **0.55** |

### The baseline here is 63.1%, not 50%

The classes are not balanced. Player 1 wins about 65% of the time in this dataset — the
`p1`/`p2` assignment in the source data is not arbitrary — so the test set is 136,785
wins to 79,876 losses. A model that ignores its inputs and always predicts "win" scores
**63.1%**.

That reframes the table considerably:

- **ANN (64%)** and **Decision Tree (65.0%)** are within ~2 points of the baseline. They
  are close to no better than a constant prediction.
- **Random Forest (70%)** beats it by ~7 points, and is the only model with a meaningful
  margin.

Accuracy alone flatters every model on this data, which is why the per-class F1 columns
matter. The ANN's loss-class F1 of 0.27 (recall 0.18) shows what the headline 64% hides:
it predicts "win" for almost everything. The Decision Tree scores lower overall than the
ANN but is far more balanced across the two classes.

Numbers for the Decision Tree, Random Forest, and ANN come from saved outputs in
[CS513Final_Models.ipynb](code/CS513Final_Models.ipynb). The AdaBoost figure is reported
in the [slide deck](powerpoint/); it is not among the notebook's saved outputs.
[parse.py](code/parse.py) also fits logistic regression, SVM, and a small Keras network,
but no results for those are recorded in this repo.

## Setup

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
pip install tensorflow keras   # only needed for parse.py
```

Unpack the dataset into `data/` at the repo root — both scripts and the notebook expect
it there:

```bash
unzip sets_cleaned.zip && mv sets_cleaned data
```

That gives you:

```
data/sets.csv             # raw Kaggle export
data/sets_cleaned.csv     # cleaned, categorical
data/sets_cleaned_num.csv # cleaned, label-encoded (what the models train on)
```

Both cleaned files are already in the zip, so you can go straight to the notebook. To
regenerate them from `sets.csv` instead, run `python code/parse_data.py`.

## Layout

| Path | What it is |
|---|---|
| [code/parse_data.py](code/parse_data.py) | Cleaning pipeline: expands the JSON `game_data` column into one row per game, derives `win_status` from the player-1/winner comparison, drops post-game and non-predictive columns, writes both cleaned CSVs. |
| [code/parse.py](code/parse.py) | Exploratory plots (win/loss balance, per-character winrate, stage frequency) plus a scratchpad of model fits. Run top to bottom; it opens a series of matplotlib windows. |
| [code/CS513Final_Models.ipynb](code/CS513Final_Models.ipynb) | The models the results above come from: Decision Tree, Random Forest, ANN, with confusion matrices and classification reports. |
| [powerpoint/](powerpoint/) | Final presentation (`.pptx` and `.pdf`). |

## Features

Eleven pre-game attributes, all label-encoded in `sets_cleaned_num.csv`:

`tournament_key`, `player_id`, `opponent_id`, `location_names`, `bracket_name`,
`bracket_order`, `set_order`, `best_of`, `stage`, `player_char`, `opponent_char`

Post-game columns (scores, set winner) are dropped during cleaning — they would leak the
outcome.
