import pandas as pd

df = pd.read_csv("data/Coal Mines Dataset india.csv")

print("Min:", df['Coal/ Lignite Production'].min())
print("Max:", df['Coal/ Lignite Production'].max())