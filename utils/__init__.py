__all__ = ["data", "nn", "sim","plot","rl"]

from gym.envs.registration import register

register(
    id='CartPoleContEnv-v0',
    entry_point='utils.rl:CartPoleContEnv',
)
