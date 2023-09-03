import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')

y = train_data['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('data/my_submission.csv', index=False)
print("Your submission was successfully saved!")
