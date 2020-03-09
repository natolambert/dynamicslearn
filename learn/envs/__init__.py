from gym.envs.registration import register

register(
    id='CartPoleContEnv-v0',
    entry_point='learn.envs.cartpole_continuous:CartPoleContEnv',
)

register(
    id='CrazyflieRigid-v0',
    entry_point='learn.envs.crazyflie_rigid:CrazyflieRigidEnv',
)
#
register(
    id='IonoRigid-v0',
    entry_point='learn.envs.ionocraft_rigid:IonocraftRigidEnv',
)
#
# register(
#     id='QuadEnv-v0',
#     entry_point='learn.envs.model_continuous:QuadEnv',
# )
#
# register(
#     id='DiscreteQuadEnv-v0',
#     entry_point='learn.envs.model_discrete:DiscreteQuadEnv',
# )
