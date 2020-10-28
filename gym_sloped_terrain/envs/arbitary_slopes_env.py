import sys, os
import numpy as np
import gym
import os
from gym import utils, spaces
import pdb
import gym_sloped_terrain.envs.walking_controller as walking_controller
import time
import math
import random
from collections import deque
import pybullet
import gym_sloped_terrain.envs.bullet_client as bullet_client
import pybullet_data
import gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076] #hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.077] #knee
RENDER_HEIGHT = 720 
RENDER_WIDTH = 960 
PI = np.pi
no_of_points = 100


def constrain_theta(theta):
	theta = np.fmod(theta, 2*no_of_points)
	if(theta < 0):
		theta = theta + 2*no_of_points
	return theta

class Stoch2Env(gym.Env):

	def __init__(self,
				 render = False,
				 on_rack = False,
				 gait = 'trot',
				 phase = [0, no_of_points, no_of_points,0],  #[FR, FL, BR, BL]
				 action_dim = 20,
				 collect_data = False,
				 end_steps = 1000,
				 IMU_Noise = False,
				 deg = 5):


		self._is_render = render
		self._on_rack = on_rack

		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		else:
			self._pybullet_client = bullet_client.BulletClient()

		self._theta = 0

		self._update_action_every = 1/50 
		self._frequency = 2.5 
		self.frequency_weight = 1
		self.termination_steps = end_steps

		#PD gains
		self._kp = 200
		self._kd = 10

		self.dt = 0.005
		self._frame_skip = 25
		self._n_steps = 0
		self._action_dim = action_dim

		self._obs_dim = 11

		self.action = np.zeros(self._action_dim)

		self._last_base_position = [0, 0, 0]
		self.last_yaw = 0
		self._distance_limit = float("inf")

		self.current_com_height = 0.243
		
		#wedge_parameters
		self.wedge_start = 0.5 
		self.wedge_halflength = 2
		self.wedge = [0, 0, 0, 0, 0]

		self.collect_data = collect_data
		if gait is 'trot':
			phase = [0, no_of_points, no_of_points, 0]
		elif gait is 'walk':
			phase = [0, no_of_points, 3*no_of_points/2 ,no_of_points/2]
		self._walkcon = walking_controller.WalkingController(gait_type=gait,
															 phase=phase)
		self.inverse = False
		self._cam_dist = 1.0
		self._cam_yaw = 0.0
		self._cam_pitch = 0.0

		self.avg_vel_per_step = 0
		self.avg_omega_per_step = 0

		self.linearV = 0
		self.angV = 0
		self.prev_vel=[0,0,0]

		self.x_f = 0
		self.y_f = 0

		self.clips=7

		self.friction = 0.6
		self.ori_history_length = 3
		self.ori_history_queue = deque([0]*3*self.ori_history_length, 
		                            maxlen=3*self.ori_history_length)#observation queue

		self.step_disp = deque([0]*100, maxlen=100)
		self.stride = 5

		self.incline_deg = deg
		self.incline_ori = 0

		self.prev_incline_vec = (0,0,1)

		self.terrain_pitch = []
		self.add_IMU_noise = IMU_Noise

		self.INIT_POSITION =[0,0,0.58]
		self.INIT_ORIENTATION = [0, 0, 0, 1]

		self.support_plane_estimated_pitch = 0
		self.support_plane_estimated_roll = 0


		self.pertub_steps = 0
		self.x_f = 0
		self.y_f = 0

		## Gym env related mandatory variables
		self._obs_dim = 3*self.ori_history_length + 2 #[r,p,y]x previous time steps, suport plane roll and pitch
		observation_high = np.array([np.pi/2] * self._obs_dim)
		observation_low = -observation_high
		self.observation_space = spaces.Box(observation_low, observation_high)

		action_high = np.array([1] * self._action_dim)
		self.action_space = spaces.Box(-action_high, action_high)
		self.hard_reset()


	def hard_reset(self):
		self._pybullet_client.resetSimulation()
		self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
		self._pybullet_client.setTimeStep(self.dt/self._frame_skip)

		self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
		self._pybullet_client.changeVisualShape(self.plane,-1,rgbaColor=[1,1,1,0.9])
		self._pybullet_client.setGravity(0, 0, -9.8)

		track_wedges = ["block1_11_11", "stairs_wedge_down", "pyramid"]
		wedge_model_path = "gym_sloped_terrain/envs/track/"

		self.wedge_halfheight = 0.2
		first_wedge = -0.5
		self.wedgePos = [first_wedge, 0, self.wedge_halfheight]
		self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
		first_wedge_path = wedge_model_path + track_wedges[0] + "/model.urdf"
		self.wedge[0] = self._pybullet_client.loadURDF(first_wedge_path, self.wedgePos, self.wedgeOrientation)

		self.wedgePos = [4 + first_wedge, -0.2, self.wedge_halfheight]
		self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
		second_wedge_path = wedge_model_path + track_wedges[1] + "/model.urdf"
		self.wedge[1] = self._pybullet_client.loadURDF(second_wedge_path, self.wedgePos, self.wedgeOrientation)

		self.wedgePos = [6.1 + first_wedge, 0.1, self.wedge_halfheight]
		self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
		third_wedge_path = wedge_model_path + track_wedges[2] + "/model.urdf"
		self.wedge[2] = self._pybullet_client.loadURDF(third_wedge_path, self.wedgePos, self.wedgeOrientation)

		self.wedgePos = [7.55 + first_wedge, -0.2, self.wedge_halfheight]
		self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
		fourth_wedge_path = wedge_model_path + track_wedges[2] + "/model.urdf"
		self.wedge[3] = self._pybullet_client.loadURDF(fourth_wedge_path, self.wedgePos, self.wedgeOrientation)

		self.SetWedgeFriction(0.7)

		model_path = 'gym_sloped_terrain/envs/robots/stoch_two_abduction_urdf/urdf/stoch_two_abduction_urdf.urdf'
		self.stoch2 = self._pybullet_client.loadURDF(model_path, self.INIT_POSITION,self.INIT_ORIENTATION)

		self._joint_name_to_id, self._motor_id_list  = self.BuildMotorIdList()

		num_legs = 4
		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=True)
		self.ResetPoseForAbd()
		if self._on_rack:
			self._pybullet_client.createConstraint(
				self.stoch2, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
				[0, 0, 0], [0, 0, 0], [0, 0, 0.3])

		self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, self.INIT_POSITION, self.INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])

		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		self.SetFootFriction(self.friction)
		self.SetLinkMass(0,0)
		self.SetLinkMass(11,0)

	def reset_standing_position(self):
		num_legs = 4
		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=10)
		self.ResetPoseForAbd()

		# Conditions for standstill
		for i in range(300):
			self._pybullet_client.stepSimulation()

		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=0)


	def reset(self):
		self._theta = 0
		self._last_base_position = [0, 0, 0]
		self.last_yaw = 0

		self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, self.INIT_POSITION, self.INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
		self.reset_standing_position()

		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		self._n_steps = 0
		return self.GetObservation()


	def apply_Ext_Force(self, x_f, y_f,link_index= 1,visulaize = False,life_time=0.01):
		force_applied = [x_f,y_f,0]
		self._pybullet_client.applyExternalForce(self.stoch2, link_index, forceObj=[x_f,y_f,0],posObj=[0,0,0],flags=self._pybullet_client.LINK_FRAME)
		f_mag = np.linalg.norm(np.array(force_applied))

		if(visulaize and f_mag != 0.0):
			point_of_force = self._pybullet_client.getLinkState(self.stoch2, link_index)[0]
			
			lam = 1/(2*f_mag)
			dummy_pt = [point_of_force[0]-lam*force_applied[0],
			 			point_of_force[1]-lam*force_applied[1],
			 			point_of_force[2]-lam*force_applied[2]]
			self._pybullet_client.addUserDebugText(str(round(f_mag,2))+" N",dummy_pt,[0.13,0.54,0.13],textSize=2,lifeTime=life_time)
			self._pybullet_client.addUserDebugLine(point_of_force,dummy_pt,[0,0,1],3,lifeTime=life_time)


	def SetLinkMass(self,link_ind,mass=0):

		if(link_ind==0):
			mass=mass+1.1
			self._pybullet_client.changeDynamics(self.stoch2, 0, mass=mass)
		elif(link_ind==11):
			mass=mass+1.1
			self._pybullet_client.changeDynamics(self.stoch2, 11, mass=mass)
		return mass
		

	def getlinkmass(self,linkind):
		m = self._pybullet_client.getDynamicsInfo(self.stoch2,linkind)
		return m[0]


	def boundYshift(self, x, y):
		if x > 0.5619:
			if y > 1/(0.5619-1)*(x-1):
				y = 1/(0.5619-1)*(x-1)
		return y

	def getYXshift(self, yx):
		y = yx[:4]
		x = yx[4:]
		for i in range(0,4):
			y[i] = self.boundYshift(abs(x[i]), y[i])
			y[i] = y[i] * 0.038
			x[i] = x[i] * 0.0418
		yx = np.concatenate([y,x])
		return yx

	def transform_action(self, action):

		action = np.clip(action, -1, 1)

		action[:4] = (action[:4] + 1)/2			# Step lengths are positive always

		action[:4] = action[:4] *2 * 0.068  	# Max steplength = 2x0.068

		action[4:8] = action[4:8] * PI/2		#PHI can be [-pi/2, pi/2]

		action[8:12] = (action[8:12]+1)/2		# el1ipse center y is positive always

		action[8:16] = self.getYXshift(action[8:16])

		action[16:20] = action[16:20]*0.035		# ellipse in and out

		action[17] = -action[17]

		action[19] = -action[19]

		return action

	def get_foot_contacts(self):

		foot_ids = [8,3,19,14]
		foot_contact_info = np.zeros(8)

		for leg in range(4):
			contact_points_with_ground = self._pybullet_client.getContactPoints(self.plane, self.stoch2, -1, foot_ids[leg])
			if len(contact_points_with_ground) > 0:
				foot_contact_info[leg] = 1

			for wedge in self.wedge:
				contact_points_with_wedge = self._pybullet_client.getContactPoints(wedge, self.stoch2, -1, foot_ids[leg])
				if len(contact_points_with_wedge) > 0:
					foot_contact_info[leg+4] = 1

		return foot_contact_info


	def step(self, action, callback=None):
		action = self.transform_action(action)
		
		self.do_simulation(action, n_frames = self._frame_skip)

		ob = self.GetObservation()
		reward = self._get_reward()
		return ob, reward, False,{}

	def CurrentVelocities(self):
		current_w = self.GetBaseAngularVelocity()[2]
		current_v = self.GetBaseLinearVelocity()
		radial_v = math.sqrt(current_v[0]**2 + current_v[1]**2)
		return radial_v, current_w


	def do_simulation(self, action, n_frames):
		'''
		Converts action parameters to corresponding motor commands with the help of a elliptical trajectory controller
		'''
		omega = 2 * no_of_points * self._frequency  
		self.action = action
		ii = 0

		leg_m_angle_cmd = self._walkcon.run_elliptical_Traj_Stoch2(self._theta,action)

		self._theta = constrain_theta(omega * self.dt + self._theta)
		
		m_angle_cmd_ext = np.array(leg_m_angle_cmd)
	
		m_vel_cmd_ext = np.zeros(12)

		force_visualizing_counter = 0

		for _ in range(n_frames):
			ii = ii + 1
			applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
			self._pybullet_client.stepSimulation()

			if self._n_steps >=self.pertub_steps and self._n_steps <= self.pertub_steps + self.stride:
				force_visualizing_counter += 1
				if(force_visualizing_counter%7==0):
					self.apply_Ext_Force(self.x_f,self.y_f,visulaize=True,life_time=0.1)
				else:
					self.apply_Ext_Force(self.x_f,self.y_f,visulaize=False)

	
		contact_info = self.get_foot_contacts()
		pos, ori = self.GetBasePosAndOrientation()

		Rot_Mat = self._pybullet_client.getMatrixFromQuaternion(ori)
		Rot_Mat = np.array(Rot_Mat)
		Rot_Mat = np.reshape(Rot_Mat,(3,3))

		plane_normal,self.support_plane_estimated_roll,self.support_plane_estimated_pitch = normal_estimator.vector_method_Stoch2(self.prev_incline_vec, contact_info, self.GetMotorAngles(), Rot_Mat)
		self.prev_incline_vec = plane_normal

		self._n_steps += 1

	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])

		base_pos, _ = self.GetBasePosAndOrientation()
		view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
				cameraTargetPosition=base_pos,
				distance=self._cam_dist,
				yaw=self._cam_yaw,
				pitch=self._cam_pitch,
				roll=0,
				upAxisIndex=2)
		proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
				fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
				nearVal=0.1, farVal=100.0)
		(_, _, px, _, _) = self._pybullet_client.getCameraImage(
				width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
				projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

		rgb_array = np.array(px).reshape(RENDER_WIDTH, RENDER_HEIGHT, 4)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def _termination(self, pos, orientation, RPY):
		done = False
		penalty = 0
		rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
		local_up = rot_mat[6:]

		if self._n_steps >= self.termination_steps:
			done = True
			penalty = 0
		else:
			if abs(RPY[0]) > math.radians(30):
				print('Oops, Robot about to fall sideways! Terminated')
				done = True
				penalty = penalty + 0.1

			if abs(RPY[1])>math.radians(35):
				print('Oops, Robot doing wheely! Terminated')
				done = True
				penalty = penalty + 0.1

			if pos[2] > 0.7:
				print('Robot was too high! Terminated')
				done = True
				penalty = penalty + 0.6

		if done and self._n_steps <= 2:
			penalty = 3

		return done, penalty

	def _get_reward(self):

		wedge_angle = self.incline_deg*PI/180
		robot_height_from_support_plane = 0.243	
		pos, ori = self.GetBasePosAndOrientation()

		RPY_orig = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.round(RPY_orig, 4)

		current_height = round(pos[2], 5)
		self.current_com_height = current_height
		standing_penalty=0
	
		desired_height = (robot_height_from_support_plane)/math.cos(wedge_angle) + math.tan(wedge_angle)*((pos[0])*math.cos(self.incline_ori)+ 0.5)

		roll_reward = np.exp(-45 * ((RPY[0]-self.support_plane_estimated_roll) ** 2))
		pitch_reward = np.exp(-45 * ((RPY[1]-self.support_plane_estimated_pitch) ** 2))
		yaw_reward = np.exp(-35 * (RPY[2] ** 2))
		height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

		x = pos[0]
		x_l = self._last_base_position[0]
		self._last_base_position = pos

		step_distance_x = (x - x_l)

		done, penalty = self._termination(pos, ori, RPY_orig)
		if done:
			reward = 0
		else:
			reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4)\
					 + round(height_reward,4) + 100 * round(step_distance_x, 4)

			# self.step_disp.append(step_distance_x)
			# if(self._n_steps>150):
			# 	if(sum(self.step_disp)<0.035):
			# 		reward = reward-3

		return reward

	def _apply_pd_control(self, motor_commands, motor_vel_commands):
		qpos_act = self.GetMotorAngles()
		qvel_act = self.GetMotorVelocities()
		applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

		applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
		applied_motor_torque = applied_motor_torque.tolist()

		for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
			self.SetMotorTorqueById(motor_id, motor_torque)
		return applied_motor_torque

	def add_noise(self, sensor_value):
		noise = np.random.normal(0, 0.04, 1)
		sensor_value = sensor_value + noise[0]
		return sensor_value

	def GetObservation(self):
		pos, ori = self.GetBasePosAndOrientation()
		RPY = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.round(RPY, 5)

		for val in RPY:
			if(self.add_IMU_noise):
				val = self.add_noise(val)
			self.ori_history_queue.append(val)

		obs = np.concatenate((self.ori_history_queue, [self.support_plane_estimated_roll,self.support_plane_estimated_pitch])).ravel()

		return obs

	def GetMotorAngles(self):
		motor_ang = [self._pybullet_client.getJointState(self.stoch2, motor_id)[0] for motor_id in self._motor_id_list]
		return motor_ang

	def GetMotorVelocities(self):
		motor_vel = [self._pybullet_client.getJointState(self.stoch2, motor_id)[1] for motor_id in self._motor_id_list]
		return motor_vel

	def GetBasePosAndOrientation(self):
		position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.stoch2))
		return position, orientation

	def GetBaseAngularVelocity(self):
		basevelocity= self._pybullet_client.getBaseVelocity(self.stoch2)
		return basevelocity[1] #world AngularVelocity vec3, list of 3 floats

	def GetBaseLinearVelocity(self):
		basevelocity= self._pybullet_client.getBaseVelocity(self.stoch2)
		return basevelocity[0] #world linear Velocity vec3, list of 3 floats

	def SetFootFriction(self, foot_friction):
		FOOT_LINK_ID = [3,8,14,19]
		for link_id in FOOT_LINK_ID:
			self._pybullet_client.changeDynamics(
			self.stoch2, link_id, lateralFriction=foot_friction)
		return foot_friction

	def SetWedgeFriction(self, friction):
		for wedge in self.wedge:
			self._pybullet_client.changeDynamics(wedge, -1, lateralFriction=friction)

	def SetMotorTorqueById(self, motor_id, torque):
		self._pybullet_client.setJointMotorControl2(
				  bodyIndex=self.stoch2,
				  jointIndex=motor_id,
				  controlMode=self._pybullet_client.TORQUE_CONTROL,
				  force=torque)
	def BuildMotorIdList(self):
		num_joints = self._pybullet_client.getNumJoints(self.stoch2)
		joint_name_to_id = {}
		for i in range(num_joints):
			joint_info = self._pybullet_client.getJointInfo(self.stoch2, i)
			joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

		#adding abduction
		MOTOR_NAMES = [ "motor_fl_upper_hip_joint",
						"motor_fl_upper_knee_joint",
						"motor_fr_upper_hip_joint",
						"motor_fr_upper_knee_joint",
						"motor_bl_upper_hip_joint",
						"motor_bl_upper_knee_joint",
						"motor_br_upper_hip_joint",
						"motor_br_upper_knee_joint",
						"motor_front_left_abd_joint",
						"motor_front_right_abd_joint",
						"motor_back_left_abd_joint",
						"motor_back_right_abd_joint"]

		motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

		return joint_name_to_id, motor_id_list

	def ResetLeg(self, leg_id, add_constraint, standstilltorque=10):
		leg_position = LEG_POSITION[leg_id]
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"],  # motor
			targetValue=0, targetVelocity=0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id[leg_position + "lower_knee_joint"],
			targetValue=0, targetVelocity=0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_" + leg_position + "upper_hip_joint"],  # motor
			targetValue=0, targetVelocity=0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id[leg_position + "lower_hip_joint"],
			targetValue=0, targetVelocity=0)

		if add_constraint:
			c = self._pybullet_client.createConstraint(
				self.stoch2, self._joint_name_to_id[leg_position + "lower_hip_joint"],
				self.stoch2, self._joint_name_to_id[leg_position + "lower_knee_joint"],
				self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
				KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

			self._pybullet_client.changeConstraint(c, maxForce=200)

		# set the upper motors to zero
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=standstilltorque)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id["motor_" + leg_position + "upper_hip_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=standstilltorque)

		# set the lower joints to zero
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id[leg_position + "lower_hip_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=0)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id[leg_position + "lower_knee_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=0)
		# set the spine motors to zero
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id["motor_front_body_spine_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.stoch2,
			jointIndex=(self._joint_name_to_id["motor_back_body_spine_joint"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0)

	def ResetPoseForAbd(self):
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_front_left_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_front_right_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_back_left_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_back_right_abd_joint"],
			targetValue = 0, targetVelocity = 0)

		#Set control mode for each motor and initial conditions
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_front_left_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_front_right_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_back_left_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_back_right_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)




