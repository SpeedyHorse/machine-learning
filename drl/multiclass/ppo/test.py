from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from time import time
import datetime
sys.path.append("/Users/toshi_pro/Documents/github-sub/machine-learning")
import flowdata
import flowenv
import pandas as pd
from sklearn.metrics import classification_report


def write_action_and_answer(episode=None, actions=None, answers=None, test=False):
    all_data = []
    class_name = flowdata.flow_data.label_info()
    if test:
        file_name = "action_and_answer_test.csv"
        report_dict = classification_report(answers, actions, labels=[0, 1, 2, 3, 4], target_names=class_name, output_dict=True, zero_division=0)
        for class_, metrics in report_dict.items():
            if class_ in class_name:
                entry = {
                    "episode": episode,
                    "class": class_,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"]
                }
                all_data.append(entry)
        df = pd.DataFrame(all_data)
        df.to_csv(file_name, mode="w", header=False)
        return
    
    file_name = "action_and_answer.csv"
    # y_true: answer, y_pred: action
    for i, (y_true, y_pred) in enumerate(zip(answers, actions)):
        report_dict = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], target_names=class_name, output_dict=True, zero_division=0)
        for class_, metrics in report_dict.items():
            if class_ in class_name:
                entry = {
                    "episode": i,
                    "class": class_,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"]
                }
                all_data.append(entry)
    
    df = pd.DataFrame(all_data)
    df.to_csv(file_name, mode="w", header=False)


data, info = flowdata.flow_data.using_multiple_data()
raw_data_test = data[1]
# 環境の作成
print("make env")
test_env = gym.make("flowenv/MultiFlow-v1", data=raw_data_test)
print("make env end")

test_model = PPO.load("ppo_no1", test_env)

count_action = []
count_answer = []
# トレーニング済みモデルでテスト
print("test start")
confusion_array = np.zeros((2, 2), dtype=np.int32)
obs, _ = test_env.reset()
for _ in range(10000):
    action, _states = test_model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    index = info["confusion_position"]

    count_action.append(info["action"])
    count_answer.append(info["answer"])

    confusion_array[index[0], index[1]] += 1
    if done:
        obs, _ = test_env.reset()

write_action_and_answer(10000, actions=count_action, answers=count_answer, test=True)