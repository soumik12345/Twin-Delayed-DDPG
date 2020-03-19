from test import Tester


# tester = Tester(
#     './Configs/BipedalWalker-v3.json',
#     './pretrained_models/bipedal_walker_v3/bipedal_walker_v3_0'
# )

# tester = Tester(
#     './Configs/LunarLanderContinuous-v2.json',
#     './pretrained_models/lunar_lander_v2/ddpg_lunarlander_v2_0'
# )

tester = Tester(
    './Configs/BipedalWalkerHardcore-v3.json',
    './pretrained_models/bipedal_walker_hardcore_v3/bipedal_walker_hardcore_v3_0'
)

print('Mean Reward:', tester.test(eval_episodes=20, render=True))
