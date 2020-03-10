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
    ## linux machine
    # df = load_data("../../data/baby-weights-dataset.csv")
    ## windows machine
    df = load_data("C:\\Users\\51606\\Desktop\\dstoolkit\\data\\baby-weights-dataset.csv")
    print("Data has been loaded!")
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("BWEIGHT", axis=1))
    X = np.insert(X,0,1,axis=1)
    y = df["BWEIGHT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    reg = gd.GradientDescent(method="sgd")
    print(reg.algorithm)
    print("***Training the model...***")
    reg.fit(X_train, y_train)
    print("***Training done!***")
    print(reg.weights)
    print(f"The number of iterations to stop: {reg.num_iter}")
    print(f"The corresponding training MSE: {reg.mse}")
    y_pred = reg.predict(X_test)
    print(f"The testing MSE: {mse(y_test,y_pred)}")

if __name__ == "__main__":
    main()
