from vrep_env import vrep_env
import os, time
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PioneerVrepEnv(vrep_env.VrepEnv):
	metadata = {'render.modes': [],}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port= 25000,
		scene_path=vrep_scenes_path+'p3dx_explore.ttt'
	):
		print(scene_path)
		vrep_env.VrepEnv.__init__(
			self,
			server_addr,
			server_port,
			scene_path,
		)

		# Settings
		self.random_start = False
		
		# All sensors
		sensor_names = ['Pioneer_p3dx_ultrasonicSensor' + str(i) for i in range(1,9)]
		# All joints
		joint_names = ['Pioneer_p3dx_leftMotor','Pioneer_p3dx_rightMotor']
		# Robot
		robot_name = 'Pioneer_p3dx'
		
		# Getting object handles
		
		# Sensors
		self.oh_sensor = list(map(self.get_object_handle, sensor_names))

		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))

		# Robot
		self.oh_robot = self.get_object_handle(robot_name)
		
		# One action per joint
		dim_act = len(self.oh_joint)
		# Multiple dimensions per shape
		dim_obs = len(self.oh_sensor)
		
		high_act =        np.ones([dim_act])
		high_obs = np.inf*np.ones([dim_obs])
		
		self.action_space      = gym.spaces.Box(-high_act, high_act)
		self.observation_space = gym.spaces.Box(-high_obs, high_obs)
		
		# Parameters
		self.joints_max_velocity = 8.0
		#self.power = 0.75
		self.power = 0.3
		
		self.seed()
		
		print('Pioneer_p3dx_VrepEnv: initialized')

	def _make_observation(self):
		"""Get observation from v-rep and stores in self.observation
		"""
		lst_o = []
		
		# Get data from ultrasonic sensor
		for i_oh in self.oh_sensor:
			#print(lst_o)
			lst_o += self.obj_read_proximity_sensor(i_oh)
		
		self.observation = np.array(lst_o).astype('float32')

	def _make_action(self, a):
		"""Send action to v-rep
		"""
		#print([0.5+self.power*float(np.clip(i_a,-1,+1)) for i_a in a])
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, 1.0+self.power*float(np.clip(i_a,-1,+1)))

	def step(self, action):
		# Clip xor Assert
		#assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		reward = -(0.5 - np.min(self.observation))**2.0
		done = (reward < -20.0)

		return self.observation, [reward], done, {}

	def reset(self):
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		# First action is random: emulate random initialization
		if self.random_start:
			factor = self.np_random.uniform(low=0, high=0.02, size=(1,))[0]
			action = self.action_space.sample()*factor
			self._make_action(action)
			self.step_simulation()
		
		self._make_observation()
		return self.observation
	
	def render(self, mode='human', close=False):
		pass

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

