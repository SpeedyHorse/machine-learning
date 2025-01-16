from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/toshi_pro/Documents/github-sub/machine-learning")
import flowdata
import flowenv

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

raw_data_train, raw_data_test = flowdata.flow_data.using_data()
# 環境の作成
print("make env")
env = make_vec_env("flowenv/Flow-v1", n_envs=4, env_kwargs={"data": raw_data_train})  # 複数環境で並列実行
print("make env end")

# A2Cエージェントの作成
print("make model")
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=1e-5,
    n_steps=2048,
    batch_size=32,
    n_epochs=20
)

for i in range(100):
    print(f"train start {i}")
    # トレーニング
    model.learn(total_timesteps=1000)
    print("train end")

    # model.save("ppo_no4")

    # model = PPO.load("ppo_no4")

    # トレーニング済みモデルでテスト
    print("test start")
    confusion_array = np.zeros((2, 2), dtype=np.int32)
    obs = env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        index = info[0]["confusion_position"]
        confusion_array[index[0], index[1]] += 1
        if done.any():
            obs = env.reset()

    # print(confusion_array)

    tp = confusion_array[0, 0]
    tn = confusion_array[1, 1]
    fp = confusion_array[0, 1]
    fn = confusion_array[1, 0]

    accuracy, precision, recall, f1, fpr = calculate_metrics(tp, tn, fp, fn)
    print(accuracy, precision, recall, f1, fpr)
    plt.figure()
    plt.bar(
        ["accuracy", "precision", "recall", "f1", "fpr"],
        [accuracy, precision, recall, f1, fpr]
    )
    plt.pause(0.1)
