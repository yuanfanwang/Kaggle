import pandas as pd
import matplotlib.pyplot as plt
import statistics

train = pd.read_csv("data/train1010.csv")

# train["wording_pred"] = train["content_pred"]
x = train["content"]
y = train["wording"]
x_pred = train["content_pred"]
y_pred = train["wording_pred"]

x_mean      = statistics.mean(x)
y_mean      = statistics.mean(y)
x_std       = statistics.stdev(x)
y_std       = statistics.stdev(y)
x_pred_mean = statistics.mean(x_pred)
y_pred_mean = statistics.mean(y_pred)
x_pred_std  = statistics.stdev(x_pred)
y_pred_std  = statistics.stdev(y_pred)

print(statistics.mean(x))
print(statistics.mean(y))
print(statistics.stdev(x))
print(statistics.stdev(y))

print("content: ", x_pred_mean)
print("wording: ", y_pred_mean)
print("contnet: ", x_pred_std)
print("wording: ", y_pred_std)

x_pred_st = [(x - x_pred_mean) / x_pred_std for x in x_pred]
y_pred_st = [(y - y_pred_mean) / y_pred_std for y in y_pred]

plt.scatter(x, y, label='answer', color='blue', marker='o', s=10)
plt.scatter(x_pred, y_pred, label='pred', color='red', marker='x', s=10)
# plt.scatter(x_pred_st, y_pred_st, label='pred', color='green', marker='*', s=10)

plt.xlabel('content')
plt.ylabel('wording')
plt.title("content and wording")
plt.show()

