import pandas as pd
import numpy as np
from flowenv.src.const import Const
from pathlib import Path


def min_max_p(p):
    min_p = p.min()
    max_p = p.max()
    return (p - min_p) / (max_p - min_p)

CONST = Const()

TRAIN_DATA_PATH = "../../DNP3_Intrusion_Detection_Dataset_Final/Training_Testing_Balanced_CSV_Files/CICFlowMeter/CICFlowMeter_Training_Balanced.csv"
TEST_DATA_PATH = "../../DNP3_Intrusion_Detection_Dataset_Final/Training_Testing_Balanced_CSV_Files/CICFlowMeter/CICFlowMeter_Testing_Balanced.csv"

TRAIN_DATA_PATH = Path(__file__).resolve().parent.joinpath(TRAIN_DATA_PATH)
TEST_DATA_PATH = Path(__file__).resolve().parent.joinpath(TEST_DATA_PATH)

def using_data():
    train_data = pd.read_csv(TRAIN_DATA_PATH).replace([np.inf, -np.inf], np.nan).dropna(how="all").dropna(how="all", axis=1)
    test_data = pd.read_csv(TEST_DATA_PATH).dropna(how="all").replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)

    train_data["Binary Label"] = train_data["Label"] == "NORMAL"
    test_data["Binary Label"] = test_data["Label"] == "NORMAL"

    for label in CONST.normalization_features:
        train_data[label] = min_max_p(train_data[label])
        test_data[label] = min_max_p(test_data[label])

    return train_data, test_data