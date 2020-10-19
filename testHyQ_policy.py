import sys, os
#sys.path.append(os.path.realpath('../'))
import gym_stoch2_sloped_terrain.envs.HyQ_pybullet_env as e
import argparse
from fabulous.color import blue,green,red,bold
import numpy as np
import math
PI = np.pi


#policy to be tested 
policy = np.load("experiments/19Oct1/iterations/best_policy.npy")
#policy = np.load("initial_policies/initial_policy_HyQ.npy")


rpy_accurate = []
rpy_noisy = []
if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--FrontMass', help='mass to be added in the first', type=float, default=0)
	parser.add_argument('--BackMass', help='mass to be added in the back', type=float, default=0)
	parser.add_argument('--FrictionCoeff', help='foot friction value to be set', type=float, default=0.6)
	parser.add_argument('--WedgeIncline', help='wedge incline degree of the wedge', type=int, default=15)
	parser.add_argument('--WedgeOrientation', help='wedge orientation degree of the wedge', type=float, default=45)
	parser.add_argument('--MotorStrength', help='maximum motor Strength to be applied', type=float, default=7.0)
	parser.add_argument('--RandomTest', help='flag to sample test values randomly ', type=bool, default=False)
	parser.add_argument('--seed', help='seed for the random sampling', type=float, default=100)
	parser.add_argument('--EpisodeLength', help='number of gait steps of a episode', type=int, default=1000)

	args = parser.parse_args()
	WedgePresent = True
	if(args.WedgeIncline == 0):
		WedgePresent = False
	
	env = e.HyQEnv(render=True, wedge=WedgePresent, downhill=False, stairs = False,seed_value=args.seed,
				      on_rack=False, gait = 'trot')
	steps = 0
	t_r = 0
	if(args.RandomTest):
		env.Set_Randomization(default=False)
	else:
		env.incline_deg = args.WedgeIncline
		env.incline_ori = math.radians(args.WedgeOrientation)


	state = env.reset()


	print (
	bold(blue("\nTest Parameters:\n")),
	green('\nWedge Inclination:'),red(env.incline_deg),
	green('\nWedge Orientation:'),red(math.degrees(args.WedgeOrientation)),
	green('\nCoeff. of friction:'),red(env.friction),
	green('\nMotor saturation torque:'),red(env.clips))

	for i_step in range(args.EpisodeLength):
		print('Roll:',math.degrees(env.support_plane_estimated_roll),
		      'Pitch:',math.degrees(env.support_plane_estimated_pitch))
		action = policy.dot(state)
		# action = [1.0,1.0,1.0,1.0,
		# 		  0.0,0.0,0.0,0.0,
		# 		  -1.0,-1.0,-1.0,-1.0,
		# 		  -1.0,-1.0,-1.0,-1.0,
		#     	  0.0,0.0,0.0,0.0 ]
		# action = [0.5,0.5,0.5,0.5,
        #                 0,0,0,0,
        #                -1,-1,-1,-1,
        #                 0.0,0.0,0.0,0.0,
        #                 1.0, 1.0, 1.0, 1.0]
		
		state, r, _, angle = env.step(action)
		
		t_r +=r

	print("Total_reward "+ str(t_r))
