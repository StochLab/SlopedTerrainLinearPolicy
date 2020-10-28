import sys, os
import gym_sloped_terrain.envs.Laikago_pybullet_env as e
import argparse
from fabulous.color import blue, green, red, bold
import numpy as np
import math
PI = np.pi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='21Oct4')
    parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=11)
    parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=0)
    parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
    parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
    parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=1000)
    parser.add_argument('--Downhill', help='should robot walk downhill?', type=bool, default=True)

    args = parser.parse_args()
    policy = np.load("experiments/"+args.PolicyDir+"/iterations/best_policy.npy")    
    WedgePresent = True
    if (args.WedgeIncline == 0):
        WedgePresent = False
    elif(args.WedgeIncline <0):
        args.WedgeIncline = -1*args.WedgeIncline
        args.Downhill = True


    env = e.LaikagoEnv(render=True, wedge=WedgePresent, downhill=args.Downhill, stairs=False, seed_value=args.seed,
                       on_rack=False, gait='trot')


    if (args.RandomTest):
        env.randomize_only_inclines(default=False)
    else:
        env.incline_deg = args.WedgeIncline
        env.incline_ori = math.radians(args.WedgeOrientation)

    state = env.reset()

    print(
        bold(blue("\nTest Parameters:\n")),
        green('\nWedge Inclination:'), red(env.incline_deg),
        green('\nWedge Orientation:'), red(math.degrees(args.WedgeOrientation)),
        green('\nCoeff. of friction:'), red(env.friction),
        green('\nMotor saturation torque:'), red(env.clips))

	# Simulation starts
    t_r = 0
    for i_step in range(args.EpisodeLength):
        action = policy.dot(state)
        state, r, _, angle = env.step(action)
        t_r += r

    print("Total_reward " + str(t_r))
