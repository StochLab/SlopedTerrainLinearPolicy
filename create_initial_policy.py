import numpy as np
import gym_stoch2_sloped_terrain.envs.stoch2_pybullet_env as e
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if(__name__ == "__main__"):


	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--policyName', help='file name of the initial policy', type=str, default='initial_policy_test')
	args = parser.parse_args()	
	env = e.Stoch2Env(render=True, wedge = True, stairs = False,on_rack=False, gait = 'trot', collect_data = False , randomization= 0)

	tuned_actions= np.array([[0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                       -1,-1,-1,-1,
                        0, 0, 0, 0],
					  [0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                        0,0,0,0,
                        1, 1, 1, 1],
					  [0.5,0.5,0.5,0.5,
                        0,0,0,0,
                       -1,-1,-1,-1,
                       -0.5,-0.5,-0.5,-0.5,
                        0.5, 0.5, 0.5, 0.5]
					  ])

	# NUmber of steps per episode
	num_of_steps = 50

	# list that tracks the states and actions
	states = []
	actions = []
	do_supervised_learning = True

	
	idx1 = [3]
	idx2 = [0,3,2]
	idx3 = [1]
	experiment_counter = 0
	
	for i in idx1:
		for j in idx2:
			for k in idx3:
				t_r = 0

				env.Set_Randomization(default=True,idx1 = i,idx2=j,idx3=k,idx0=0,idx11=0)
				#print("incline:",env.incline_deg)

				cstate = env.reset()
				roll = 0
				pitch = 0
				
				for ii in np.arange(0,num_of_steps):
					cstate, r, _, info = env.step(tuned_actions[experiment_counter])
					t_r +=r
					states.append(cstate)
					actions.append(tuned_actions[experiment_counter])
				experiment_counter = experiment_counter +1
				print("Returns of the experiment:",t_r)
	

	if(do_supervised_learning):
		model = LinearRegression(fit_intercept = False)
		states = np.array(states)
		actions = np.array(actions)

		#train
		print("Shape_X_Labels:",states.shape,"Shape_Y_Labels:",actions.shape)
		model.fit(states,actions)
		action_pred= model.predict(states)
		
		#test
		print('Mean squared error:', mean_squared_error(actions, action_pred))
		res = np.array(model.coef_)
		np.save("./initial_policies/"+args.policyName+".npy", res)