## Homeworks for ITMO course "ML for scientific data analysis" (feb 2024)

_Study project to work with some new (for me) python packages and to improve some data analysis and data processing skills._

Data was taken [from kaggle](https://www.kaggle.com/datasets/beaver68/cars-dataset-in-russia): _Dataset of cars in Russia_

**Task:** to predict cars price based on their characteristics

Python version: `3.8.10`

### Homework 1

Location: `hw01-eda&baseline/`

<ol>
  <li>EDA was conducted (`eda.ipynb`)</li>
  <li>Some models were tried as a baseline (`baseline.ipynb`)</li>
    <ul>
      <li>`DummyRegressor`</li>
        <ul>
          <li>predicts median (by dataset) of price value for every input</li>
          <li>`MAE~1.00` which means that by average model prediction is wrong by `1m RUB`!</li> 
        </ul>
      <li>`RandomForestRegressor`</li>
        <ul>
          <li>`MAE~0.262` which means that by average model prediction is wrong by `260k RUB`!</li>
        </ul>
    </ul>
</ol>

### Homework 2

Location: `hw02-project/`

<ol>
  <li>Hyperparameters optimization of `RandomForest` with `optuna` (`optuned-forest.ipynb`)</li>
    <ul>
      <li>cross validation was too _expensive_ to use during the `RandomForest` optimization process: 50 trials (without `cross_validate`) takes ~2h on CPU, results of trials were saved in `optuna/01_randomforest_optuna_res.csv`</li>
      <li>_best regressor_ has resulted in `MAE~0.245` which is a bit better in contrast to the first try with `RandomForest`</li>
      <li>_feature importance_ analysis showed, that the most important featuresfor `price` prediction are (by descending of importance): `torque`, `year`, `transmissions` and `mileage`</li>
    </ul>
  <li>On the next step of the study `CatBoost` package was tried (`catboost.ipynb`) since the `RandomForest` optimization did not give tangible results</li>
    <ul>
      <li>the library turned out to be quite easy to use: all we need (for a quick start) is to specify `iterations`, `loss_function` and `cat_features` parameters for `CatBoostRegressor`</li>
      <li>`MAE~0.182` which is on `52%` better than the `RandomForest` result</li>
      <li>I have tried to train regressor on cars that costs less than some `threshold` (in order not to take into account very expensive cars, which number was too low in the dataset), but despite the fact that MAE was falling, MAPE remained at the same level (`~16%`)</li>
      <li>_feature importance_ analysis showed, that the most important featuresfor `price` prediction are (by descending of importance): `mileage`, `torque`, `year` and `consumption`</li>
    </ul>
</ol>

#### Results

Both models (`RandomFOrestRegressor` and `CatBoostRegressor`) have choosed quite logic (according to life experience) features to predict the `price` of cars. But _in my opinion_ the `CatBoost` model's top-3 important features look more believable. Also `CatBoost` model resulted in a better target metrics value (`MAE~0.182` or `MAPE~16%`, which means that average prediction distincts by 16% from the true value) which makes it the winner (among all tried models) for that task.

Unfortunely, I had no time to try other models (for example simple _fully-connected neural network_) and I focused on `CatBoost`, which I never used before.

Also I wanted to try some _unsupervised_ solutions to clasterise the data and to obtain (for example) `country`-clustered dataset. Maybe I'll try it in another tasks:)
