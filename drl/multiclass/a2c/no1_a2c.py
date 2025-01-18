from stable_baselines3 import A2C
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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp != 0 else -1
    recall = tp / (tp + fn) if tp + fn != 0 else -1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn != 0 else 0.0

    if precision < 0:
        precision = 0.0
    if recall < 0:
        recall = 0.0
    return accuracy, precision, recall, f1, fpr

def write_action_and_answer(actions=None, answers=None, test=False):
    if test:
        file_name = "action_and_answer_test.csv"
    else:
        file_name = "action_and_answer.csv"

    all_data = []
    
    class_name = flowdata.flow_data.label_info()
    # y_true: answer, y_pred: action
    for i, (y_true, y_pred) in enumerate(zip(answers, actions)):
        report_dict = classification_report(y_true, y_pred, target_names=class_name, output_dict=True)
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
raw_data_train = data[0]
raw_data_test = data[1]
# 環境の作成
print("make env")
env = make_vec_env("flowenv/MultiFlow-v1", n_envs=4, env_kwargs={"data": raw_data_train})  # 複数環境で並列実行
test_env = gym.make("flowenv/MultiFlow-v1", data=raw_data_test)
print("make env end")

# A2Cエージェントの作成
print("make model")
model = A2C(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.0001,
    n_steps=30,
    device="cpu"
)

n_outputs = test_env.action_space.n
count_action = []
count_answer = []
test_episode_actions = []
test_episode_answers = []

for i in range(10):
    # トレーニング
    print("tr_s:1 =>", end="")
    model.learn(total_timesteps=100000)
    print("tr_e")

model.save("no1_a2c")

models = A2C.load("no1_a2c", test_env)

# トレーニング済みモデルでテスト
print("te_s")
confusion_array = np.zeros((2, 2), dtype=np.int32)
obs, _ = test_env.reset()
for _ in range(10000):
    action, _states = models.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    index = info["confusion_position"]

    count_action.append(info["action"])
    count_answer.append(info["answer"])

    confusion_array[index[0], index[1]] += 1
    if done:
        obs, _ = test_env.reset()
        test_episode_actions.append(count_action)
        test_episode_answers.append(count_answer)
        count_action = []
        count_answer = []

write_action_and_answer(test_episode_actions, test_episode_answers, test=True)
# print(confusion_array)
tp = confusion_array[0, 0]
tn = confusion_array[1, 1]
fp = confusion_array[0, 1]
fn = confusion_array[1, 0]

accuracy, precision, recall, f1, fpr = calculate_metrics(tp, tn, fp, fn)
print(f"{i:3}: {accuracy:.6}, {precision:.6}, {recall:.6}, {f1:.6}, {fpr:.6}")
