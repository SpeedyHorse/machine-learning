from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("/Users/toshi_pro/Documents/github-sub/machine-learning")
import flowdata
import flowenv

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

raw_data_train, raw_data_test = flowdata.flow_data.using_data()
# 環境の作成
print("make env")
env = make_vec_env("flowenv/Flow-v1", n_envs=4, env_kwargs={"data": raw_data_train})  # 複数環境で並列実行
test_env = gym.make("flowenv/Flow-v1", data=raw_data_test)
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

for i in range(10):
    # トレーニング
    print("tr_s=>", end="")
    model.learn(total_timesteps=500000)
    print("tr_e=>", end="")

    model.save("a2c_no5")

    models = A2C.load("a2c_no5", test_env)
    
    # トレーニング済みモデルでテスト
    print("te_s")
    confusion_array = np.zeros((2, 2), dtype=np.int32)
    obs, _ = test_env.reset()
    for _ in range(10000):
        action, _states = models.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        index = info["confusion_position"]
        confusion_array[index[0], index[1]] += 1
        if done:
            obs, _ = test_env.reset()

    # print(confusion_array)

    tp = confusion_array[0, 0]
    tn = confusion_array[1, 1]
    fp = confusion_array[0, 1]
    fn = confusion_array[1, 0]

    accuracy, precision, recall, f1, fpr = calculate_metrics(tp, tn, fp, fn)
    print(f"{i:3}: {accuracy:.6}, {precision:.6}, {recall:.6}, {f1:.6}, {fpr:.6}")
else:
    plt.figure()
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    with open("test_result.csv", "w") as f:
        f.write("accuracy,precision,recall,f1,fpr\n")
        f.write(f"{accuracy},{precision},{recall},{f1},{fpr}\n")

    plt.title("Result")
    plt.bar(
        ["accuracy", "precision", "recall", "f1", "fpr"],
        [accuracy, precision, recall, f1, fpr]
    )
    plt.grid()

    plt.show()
