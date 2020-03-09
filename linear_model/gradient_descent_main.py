import GradientDescent as gd
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

def main():
    df = load_data("../../data/baby-weights-dataset.csv")
    print("Data has been loaded!")
    reg = gd.GradientDescent(method="batch")

    print(reg.algorithm)

if __name__ == "__main__":
    main()
