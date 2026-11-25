import pandas as pd

df = pd.read_csv("dataset/Hindi_train.csv")
df["source"] = df["source"].apply(lambda sentence: len(sentence.split()))
df["translated"] = df["translated"].apply(lambda sentence: len(sentence.split()))
print(df.describe())
print(df[["source", "translated"]].quantile([0.9, 0.95, 0.99]))
import matplotlib.pyplot as plt
df.hist(bins=50)
plt.show()
