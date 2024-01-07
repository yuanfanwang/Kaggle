import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

params = {
    'objective': 'regression',
    'metric': 'rmse',
}

Local = True

if Local:
    data_path = "./data/"
else:
    data_path = "/kaggle/input/predict-energy-behavior-of-prosumers/"


train_df  = pl.read_csv(data_path + 'train.csv', dtypes={'datetime': pl.Datetime()})
client_df = pl.read_csv(data_path + 'client.csv', dtypes={'date': pl.Datetime()})
test_df   = pl.read_csv(data_path + 'example_test_files/test.csv')
test_client_df = pl.read_csv(data_path + 'example_test_files/client.csv')

train_df = train_df.join(client_df, on=["county", "is_business", "product_type", "data_block_id"])
test_df = test_df.join(test_client_df, on=["county", "is_business", "product_type", "data_block_id"])

# print(train_df['datetime'].dtype, train_df['datetime'][0])
# print(train_df['date'].dtype, train_df['date'][0])

# for col in train_df.columns:
#     # print(col, train_df[col].dtype)
#     flag = False
#     for raw in train_df[col]:
#         if not flag: 
#             if raw == None:
#                 print(col, raw)
#                 flag = True

# fill None with 0
train_df = train_df.fill_null(0)

train_df = train_df.to_pandas()
test_df = test_df.to_pandas()

features = ['county',
            'is_business',
            'product_type',
            'is_consumption',
            #'datetime',
            'data_block_id',
            'eic_count',
            'installed_capacity',
            #'date',
]
target = 'target'
prediction_unit_id = 'prediction_unit_id'

# GroupKfold settings
n_splits = 4
group_kfold = GroupKFold(n_splits=n_splits)

# Parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'mae',  # Use Mean Absolute Error (MAE)
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Cross-validation
models = []
preds = []
trues = []

for fold, (train_idx, val_idx) in enumerate(group_kfold.split(train_df, train_df[target], groups=train_df[prediction_unit_id])):
    print(f"Training on fold {fold + 1}")

    # Create datasets

    # could be write like this:
    # X_train_cv = train.filter(pl.col("fold") != fold).select(feature_cols).to_pandas()
    # y_train_cv = train.filter(pl.col("fold") != fold).select("target").to_pandas()
    train_data = lgb.Dataset(train_df[features].iloc[train_idx], label=train_df[target].iloc[train_idx].to_numpy())
    val_data = lgb.Dataset(train_df[features].iloc[val_idx], label=train_df[target].iloc[val_idx].to_numpy(), reference=train_data)

    # Train the model
    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[val_data], early_stopping_rounds=10, verbose_eval=False)

    # Validation data predictions and evaluation
    y_pred = bst.predict(train_df[features].iloc[val_idx])
    trues.extend(train_df[target].iloc[val_idx].to_numpy())
    preds.extend(y_pred)
    mae = mean_absolute_error(train_df[target].iloc[val_idx].to_numpy(), y_pred)
    print(f'Fold {fold + 1} MAE: {mae}')
    fold_scores.append(mae)

    models.append(bst)

mae = mean_absolute_error(trues, preds)
print(f"cv MAE: {mae}")

# Test data predictions
test_predictions = bst.predict(test_df[features])

# Add predicted_target column to the test data
test_df['target'] = test_predictions

test_df = test_df[['row_id', 'data_block_id', 'target']]

# output submission.csv
test_df.to_csv('submission.csv', index=False)
