import os, glob
import pandas as pd

CURRENT_DIR = "./csv_captures1/mitm/"


def rewrite_csv(file):
    df = pd.read_csv(file)
    df = df.drop(columns=["Label"])
    df["Label"] = "MITM"
    df.to_csv(file, index=False)


def main():
    csvfiles = glob.glob(CURRENT_DIR + "*.csv")
    for file in csvfiles:
        rewrite_csv(file)
        print(f"Rewritten {file} successfully")


if __name__ == "__main__":
    main()
