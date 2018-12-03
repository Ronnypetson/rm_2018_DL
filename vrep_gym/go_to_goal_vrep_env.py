from vrep_env import vrep_env
import os, time
vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy
import math
import random
from decimal import *
from util import *
import time

class GoalPioneerVrepEnv(vrep_env.VrepEnv):
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
		# Wheels and caster
		wheel_names = ['Pioneer_p3dx_leftWheel', 'Pioneer_p3dx_rightWheel']\
						+ ['Pioneer_p3dx_caster_freeJoint1', 'Pioneer_p3dx_caster_freeJoint2', 'Pioneer_p3dx_caster_link', 'Pioneer_p3dx_caster_wheel', 'Pioneer_p3dx_caster_wheel_visible']
		# Robot
		robot_name = 'Pioneer_p3dx'
		
		# Getting object handles
		
		# Robot
		self.oh_robot = self.get_object_handle(robot_name)

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


		self.alvoh = self.get_object_handle('Plane')

		a = [0,1]#[-1,-0.5,0.5,1]
		b = [0,1]#[-1,-0.5,0.5,1]
		velC = 0.1
		#self.actions = [[i,j] for i in a for j in b]
		self.actions = [[1,0.2],[0.2,1],[1,1]]

		# One action per joint
		dim_act = len(self.actions)#len(self.oh_joint) * 2
		# Multiple dimensions per shape
		dim_obs = 3#len(self.oh_sensor)
		
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

		time.sleep(1)

		self.obj_set_velocity(self.oh_joint[0], 0)
		self.obj_set_velocity(self.oh_joint[1], 0)

		time.sleep(1)

		posicaoRobo = self.obj_get_position(self.oh_robot)
		posicaoAlvo = self.obj_get_position(self.alvoh)
		anguloRobo = self.obj_get_orientation(self.oh_robot)[2]
		posicaoRobo = [ float(Decimal("%.2f" % elem)) for elem in posicaoRobo ]
		posicaoAlvo = [ float(Decimal("%.2f" % elem)) for elem in posicaoAlvo ]
		anguloRobo = float(Decimal("%.2f" % anguloRobo))
		#print("a:", posicaoRobo, " ",posicaoAlvo, " ", anguloRobo)
		#self.observation = np.array(posicaoRobo + posicaoAlvo + anguloRobo).astype('float32')
		posRel = np.array(posicaoAlvo) - np.array(posicaoRobo)
		self.observation = np.array([posRel[0], posRel[1], anguloRobo])

	def _make_action(self, a):
		acao = self.actions[a]

		self.obj_set_velocity(self.oh_joint[0], acao[0]*2)
		self.obj_set_velocity(self.oh_joint[1], acao[1]*2)
		time.sleep(0.1)
		self.obj_set_velocity(self.oh_joint[0], acao[0]*2)
		self.obj_set_velocity(self.oh_joint[1], acao[1]*2)
		#(ang, linear) = self.obj_get_velocity(self.oh_robot)
		#print(ang)

		"""Send action to v-rep
		"""
		#print([0.5+self.power*float(np.clip(i_a,-1,+1)) for i_a in a])
		# Para o dqn
		'''if not isinstance(a, list):
			a_ = np.zeros(self.action_space.shape[0])
			a_[a] = 1.0
			a = a_
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, 2.0+self.power*float(np.clip(i_a,-1,+1)))'''

	def step(self, action):
		# Clip xor Assert
		#assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		
		posicaoRoboAnt = self.obj_get_position(self.oh_robot)
		posicaoAlvoAnt = self.obj_get_position(self.alvoh)
		anguloRoboAnt = self.obj_get_orientation(self.oh_robot)[2]

		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()

		self.obsAnterior = copy.copy(self.observation)

		# Observe
		self._make_observation()

		
		posicaoRobo = self.obj_get_position(self.oh_robot)
		posicaoAlvo = self.obj_get_position(self.alvoh)
		anguloRobo = self.obj_get_orientation(self.oh_robot)[2]

		# Reward
		distanciaAnt = math.sqrt( ((self.obsAnterior[0])**2) + ( (self.obsAnterior[1])**2) )
		#distancia = math.sqrt( ((self.observation[3]-self.observation[0])**2) + ( (self.observation[4]-self.observation[1])**2) )
		distancia = math.sqrt( ((self.observation[0])**2) + ( (self.observation[1])**2) )
		erroAngulo = math.degrees(abs(self.calcErrorA(posicaoRobo[0],posicaoRobo[1],anguloRobo,posicaoAlvo[0],posicaoAlvo[1])))
		erroAnguloAnt = math.degrees(abs(self.calcErrorA(posicaoRoboAnt[0],posicaoRoboAnt[1],anguloRoboAnt,posicaoAlvoAnt[0],posicaoAlvoAnt[1])))
		erroAngulo = 1 if erroAngulo < 1 else erroAngulo
		erroAnguloAnt = 1 if erroAnguloAnt < 1 else erroAnguloAnt
		#print("D:", (100/distancia), "\t", "E:", (erroAngulo/180))
		#reward = (20 - distancia) - ((erroAngulo/180)*2)
		#print("D:", , " \t E:", (erroAnguloAnt/180) - (erroAngulo/180))
		reward = (distanciaAnt - distancia) + ( (erroAnguloAnt/180) - (erroAngulo/180) )

		#print("Obs: ", self.observation, "\t Act:", self.actions[action], "\t R:", reward)

		lst_o = []
		for i_oh in self.oh_sensor:
			lst_o += self.obj_read_proximity_sensor(i_oh)
		sensores = np.array(lst_o).astype('float32')
		min_dist= np.min(sensores)
		done = (min_dist < 0.1) or (distancia < 0.3)# (reward < -0.18) or 
		print(distancia)
		#print(reward)

		return self.observation, reward, done, {} # [reward] for actor-critic, reward for dqn

	def reset(self):
		if self.sim_running:
			#self.stop_simulation()
			# Reset position
			posicoes = [[-2.5,-0.175,0.13868],[2.475,0.5,0.13868],[-0.15,-3.875,0.13868]]
			p_index = np.random.randint(len(posicoes))
			in_tuple = [ [], posicoes[p_index], [], bytearray() ]
			self.call_childscript_function('Pioneer_p3dx','reset_function',in_tuple)
			'''posicoes = [[-2.5,-0.175,0.13868],[2.475,0.5,0.13868],[-0.15,-3.875,0.13868]]
			p_index = np.random.randint(len(posicoes))
			ori = self.obj_get_orientation(self.oh_robot)
			ori[2] = random.uniform(0, 2*np.pi)
			self.pause_comunication(True)
			self.obj_set_position(self.oh_robot,posicoes[p_index])
			self.obj_set_orientation(self.oh_robot,ori)
			for sh, ip, io in zip(self.oh_sensor,self.ip_sensor,self.io_sensor):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
			for sh, ip, io in zip(self.oh_joint,self.ip_joint,self.io_joint):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
			for sh, ip, io in zip(self.oh_wheel,self.ip_wheel,self.io_wheel):
				self.obj_set_position(sh,ip,self.oh_robot)
				self.obj_set_orientation(sh,io,self.oh_robot)
			self.pause_comunication(False)'''
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

	def normAngle(self, theta):
		return theta % (2*math.pi)

	def calcErrorA(self,roboX,roboY,roboA,goalX,goalY):
		roboA = self.normAngle(roboA)
		y1 = roboY-goalY
		x1 = roboX-goalX
		angGoal = self.normAngle(math.pi + math.atan2(y1, x1))
		return math.atan2(math.sin(angGoal-roboA), math.cos(angGoal-roboA))


