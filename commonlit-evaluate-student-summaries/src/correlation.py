import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats

summaries_train = pd.read_csv('data/summaries_train.csv')

x = summaries_train['content']
y = summaries_train['wording']

correlation, _ = stats.pearsonr(x, y)
print(correlation)

plt.figure(figsize=(10, 10))
sns.scatterplot(x=x, y=y)

plt.xlabel('content')
plt.ylabel('wording')

plt.show()