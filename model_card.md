# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is an XGBoost classifier from the XGBOOST library which is an open-source software library that provides a regularizing gradient boosting framework

## Intended Use

The model predicts a salary based on multiple features

## Training Data

The data is downloaded from https://archive.ics.uci.edu/ml/datasets/census+income.
Unknown values were removed "?" + simple process of encoding was performed

## Evaluation Data

The same source of data as training data, the evaluation takes 20% of the initial data and isn't used in the training phase 

## Metrics

The model has precision of 77.82% recall of 66.37% and fbeta of 71.64%

## Ethical Considerations

Some features like gender and race can cause bias toward specific genders and races

## Caveats and Recommendations

Some features have high correlation with other features like the two education features ( categorical and numerical), a potential improvement can be to one feature for each highly correlating pair of features which may help the model converge better in the same execution time