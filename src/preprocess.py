import pandas as pd

def load_data():
    df = pd.read_csv("data/Coal Mines Dataset india.csv")
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    # Drop column if exists
    if 'Type of Mine' in df.columns:
        df.drop(columns=['Type of Mine'], inplace=True)

    # Reorder columns (target first)
    cols = list(df.columns)
    cols[0], cols[2] = cols[2], cols[0]
    df = df[cols]

    # Convert categorical → numeric
    df = pd.get_dummies(df, drop_first=True)

    return df

def split_data(df):
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return X, y