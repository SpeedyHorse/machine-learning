from .src import flow_train, flow_test, const
from gymnasium.envs.registration import register

register(
    id='flowenv/FlowTrain-v0',
    entry_point='flowenv.src.flow_train:FlowTrainEnv',
)

register(
    id='flowenv/FlowTest-v0',
    entry_point='flowenv.src.flow_test:FlowTestEnv',
)