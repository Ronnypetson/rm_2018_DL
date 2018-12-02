from vrep_env import vrep_env
import os, time
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

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
		self.target_point = [0.0,0.0]

		# All sensors
		sensor_names = ['Pioneer_p3dx_ultrasonicSensor' + str(i) for i in range(1,9)]
		# All joints
		joint_names = ['Pioneer_p3dx_leftMotor','Pioneer_p3dx_rightMotor']
		# Wheels and caster
		wheel_names = ['Pioneer_p3dx_leftWheel', 'Pioneer_p3dx_rightWheel']\
						+ ['Pioneer_p3dx_caster_freeJoint1', 'Pioneer_p3dx_caster_freeJoint2', 'Pioneer_p3dx_caster_link', 'Pioneer_p3dx_caster_wheel', 'Pioneer_p3dx_caster_wheel_visible', 'Pioneer_p3dx_caster_link_visible']\
						+ ['Pioneer_p3dx_leftWheel_visible', 'Pioneer_p3dx_rightWheel_visible']\
						+ ['Pioneer_p3dx_connection' + str(i) for i in range(1,11)]
		# Robot
		robot_name = 'Pioneer_p3dx'
		
		# Getting object handles
		
		# Robot
		self.oh_robot = self.get_object_handle(robot_name)
		self.ip_robot = [self.obj_get_position(self.oh_robot),self.obj_get_position(self.oh_robot)]
		self.io_robot = [self.obj_get_orientation(self.oh_robot),self.obj_get_orientation(self.oh_robot)]
		self.io_robot[-1][2] = np.pi/2.0

		# Sensors
		self.oh_sensor = list(map(self.get_object_handle, sensor_names))
		self.ip_sensor = list(map(lambda x: self.obj_get_position(x,self.oh_robot), self.oh_sensor))
		self.io_sensor = list(map(lambda x: self.obj_get_orientation(x,self.oh_robot), self.oh_sensor))

		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		self.ip_joint = list(map(lambda x: self.obj_get_position(x,self.oh_robot), self.oh_joint))
		self.io_joint = list(map(lambda x: self.obj_get_orientation(x,self.oh_robot), self.oh_joint))

		# Wheels
		self.oh_wheel = list(map(self.get_object_handle, wheel_names))
		self.ip_wheel = list(map(lambda x: self.obj_get_position(x,self.oh_robot), self.oh_wheel))
		self.io_wheel = list(map(lambda x: self.obj_get_orientation(x,self.oh_robot), self.oh_wheel))

		# One action per joint
		dim_act = len(self.oh_joint)
		# Multiple dimensions per shape
		dim_obs = len(self.oh_sensor) + 5 # obj_pos, obj_or, target

		high_act =        np.ones([dim_act])
		high_obs = np.inf*np.ones([dim_obs])
		
		self.action_space      = gym.spaces.Box(-high_act, high_act)
		self.observation_space = gym.spaces.Box(-high_obs, high_obs)
		
		# Parameters
		self.joints_max_velocity = 8.0
		#self.power = 0.75
		self.power = 0.6
		
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

		lst_o += self.obj_get_position(self.oh_robot)[:-1]
		lst_o += [self.obj_get_orientation(self.oh_robot)[2]]
		lst_o += self.target_point

		self.observation = np.array(lst_o).astype('float32')

	def _make_action(self, a):
		"""Send action to v-rep
		"""
		#print([0.5+self.power*float(np.clip(i_a,-1,+1)) for i_a in a])
		# Para o dqn
		if not isinstance(a, list):
			a_ = np.zeros(self.action_space.shape[0])
			a_[a] = 1.0
			a = a_
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, 2.0+self.power*float(np.clip(i_a,-1,+1)))

	def ciclo_trig(self,ang):
		return (ang+2*math.pi)%(2*math.pi)

	def unit_vector(self,vector):
		return vector / np.linalg.norm(vector)

	def angle(self,v1,v2):
		v1_u = self.unit_vector(v1)
		v2_u = self.unit_vector(v2)
		theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
		rot_theta = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
		theta = theta*180.0/math.pi
		theta = (360.0+theta)%360.0
		if abs(np.dot(np.dot(rot_theta,v2_u),v1_u)-1.0) > 1e-2:
			theta = 360.0 - theta
		return theta

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
		#print(self.observation[3:5])
		min_dist = min(1.0,np.min(self.observation[:8]))
		cur_pos = np.array(self.obj_get_position(self.oh_robot)[:-1])
		cur_angle = self.ciclo_trig(self.obj_get_orientation(self.oh_robot)[2])
		ang_dist = self.angle(self.target_point-cur_pos,np.array([math.cos(cur_angle),math.sin(cur_angle)]))
		lin_dist = np.linalg.norm(cur_pos-self.target_point)

		reward = -abs(0.5-min_dist) + np.min(self.observation[3:5])\
							- 30*lin_dist - min(ang_dist,360.0-ang_dist)
		print(reward)
		done = (min_dist < 0.1) or (lin_dist > 8.0) #(min_dist > 0.75)

		return self.observation, reward, done, {} # [reward] for actor-critic, reward for dqn

	def reset(self):
		if self.sim_running:
			#self.stop_simulation()
			# Reset position
			p_index = np.random.randint(len(self.ip_robot))
			self.obj_set_position(self.oh_robot,self.ip_robot[p_index])
			self.obj_set_orientation(self.oh_robot,self.io_robot[p_index])
			for sh, ip, io in zip(self.oh_sensor,self.ip_sensor,self.io_sensor):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
			for sh, ip, io in zip(self.oh_joint,self.ip_joint,self.io_joint):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
			for sh, ip, io in zip(self.oh_wheel,self.ip_wheel,self.io_wheel):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
		else:
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


