import GradientDescent as gd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.drop(["ID", "HISPMOM", "HISPDAD"], axis=1, inplace=True)
    return df

def main():
    df = load_data("C:\\Users\\51606\\Desktop\\dstoolkit\\data\\baby-weights-dataset.csv")
    print("Data has been loaded!")
    reg = gd.GradientDescent(method="minibatch")
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("BWEIGHT", axis=1))
    X = np.insert(X,0,1,axis=1)
    y = df["BWEIGHT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    reg.fit(X_train, y_train)
    print(reg.weights)
    print(reg.mse)
    y_pred = reg.predict(X_test)
    print(mse(y_test,y_pred))
    print(reg.algorithm)

if __name__ == "__main__":
    main()
