import sys, os
#sys.path.append(os.path.realpath('../'))
import gym_sloped_terrain.envs.arbitary_slopes_env as e
import argparse
from fabulous.color import blue,green,red,bold
import gym
import pybullet as p
import numpy as np
import time
import math
PI = np.pi



if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--PolicyDir', help='directory of the policy to be tested', type=str, default='23July3')

	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=10000)

	args = parser.parse_args()
	policy = np.load("experiments/"+args.PolicyDir+"/iterations/best_policy.npy")


	env = e.Stoch2Env(render=True, on_rack=False, gait = 'trot')
	steps = 0
	t_r = 0
	state = env.reset()
	for i_step in range(args.EpisodeLength):
		action = policy.dot(state)
		state, r, _, angle = env.step(action)
		t_r +=r

		#Modify camera angle
		if i_step > 3500:
			yaw_cam = 50
		else:
			yaw_cam = -20

		if (i_step % 300 == 0):
			env._pybullet_client.resetDebugVisualizerCamera(1.3, yaw_cam, -20, env.GetBasePosAndOrientation()[0])

	print("Total_reward "+ str(t_r))
