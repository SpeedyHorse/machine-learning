import pandas as pd
import numpy as np
from flowenv.src.const import Const
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTENC


def min_max_p(p):
    min_p = p.min()
    max_p = p.max()
    return (p - min_p) / (max_p - min_p)


CONST = Const()
ATTACK_LABELS = CONST.attack_labels

TRAIN_DATA_PATH = "../../DNP3_Intrusion_Detection_Dataset_Final/Training_Testing_Balanced_CSV_Files/CICFlowMeter/CICFlowMeter_Training_Balanced.csv"
TEST_DATA_PATH = "../../DNP3_Intrusion_Detection_Dataset_Final/Training_Testing_Balanced_CSV_Files/CICFlowMeter/CICFlowMeter_Testing_Balanced.csv"

TRAIN_DATA_PATH = Path(__file__).resolve().parent.joinpath(TRAIN_DATA_PATH)
TEST_DATA_PATH = Path(__file__).resolve().parent.joinpath(TEST_DATA_PATH)

CATEGORICAL_FEATURES = ["Dst Port", "Protocol"]


def _read_data(binarize=False, balance=False):
    train_data = pd.read_csv(TRAIN_DATA_PATH).replace([np.inf, -np.inf], np.nan).dropna(how="any").dropna(how="all", axis=1)
    test_data = pd.read_csv(TEST_DATA_PATH).replace([np.inf, -np.inf], np.nan).dropna(how="any").dropna(how="all", axis=1)

    # unique
    targets = ""
    if binarize:
        train_data["Binary Label"] = train_data["Label"].apply(lambda label: label in ATTACK_LABELS)
        test_data["Binary Label"] = test_data["Label"].apply(lambda label: label in ATTACK_LABELS)
        targets = "Binary Label"
    else:
        train_data["Label Index"] = train_data["Label"].apply(lambda x: 0 if x not in ATTACK_LABELS else ATTACK_LABELS.index(x) + 1)
        test_data["Label Index"] = test_data["Label"].apply(lambda x: 0 if x not in ATTACK_LABELS else ATTACK_LABELS.index(x) + 1)
        targets = "Label Index"
    # --------
    
    train_data = train_data.filter(items=CONST.features_labels + [targets])
    test_data = test_data.filter(items=CONST.features_labels + [targets])

    conbine_data = pd.concat([train_data, test_data], ignore_index=True)
    ohe = OneHotEncoder(sparse_output=False)
    conbine_data_ohe = ohe.fit_transform(conbine_data[CATEGORICAL_FEATURES])
    conbine_data_ohe = pd.DataFrame(conbine_data_ohe, columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    conbine_data = pd.concat([conbine_data.drop(columns=CATEGORICAL_FEATURES), conbine_data_ohe], axis=1)

    train_data = conbine_data.iloc[:len(train_data) - 1]
    test_data = conbine_data.iloc[len(train_data):]

    # train_data_ohe = ohe.fit_transform(train_data[CATEGORICAL_FEATURES])
    # train_data_ohe = pd.DataFrame(train_data_ohe, columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    # train_data = pd.concat([train_data.drop(columns=CATEGORICAL_FEATURES), train_data_ohe], axis=1)
    train_smotenc_columns = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    # test_data_ohe = ohe.fit_transform(test_data[CATEGORICAL_FEATURES])
    # test_data_ohe = pd.DataFrame(test_data_ohe, columns=ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    # test_data = pd.concat([test_data.drop(columns=CATEGORICAL_FEATURES), test_data_ohe], axis=1)

    for label in CONST.normalization_features:
        train_data.loc[:, label] = min_max_p(train_data[label])
        test_data.loc[:, label] = min_max_p(test_data[label])
    
    train_data = train_data.dropna(how="any")
    test_data = test_data.dropna(how="any")
    
    if balance:
        X_train = train_data.drop(columns=[targets])
        y_train = train_data[targets]

        smote = SMOTENC(
            categorical_features=[X_train.columns.get_loc(label) for label in train_smotenc_columns], 
            random_state=42,
            k_neighbors=3
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)

        columns_name = list(train_data.columns)
        X_resampled = pd.DataFrame(X_train)
        y_resampled = pd.Series(y_train, name=targets)

        resampled_train_data = pd.concat([X_resampled, y_resampled], axis=1)

        return resampled_train_data, test_data
    else:
        return train_data, test_data


def using_nonbalanced_data():
    return _read_data(binarize=True)

def using_data():
    return _read_data(binarize=True, balance=True)


def using_multiple_data():
    index_info = ["Normal"] + ATTACK_LABELS

    return _read_data(binarize=False, balance=True), index_info