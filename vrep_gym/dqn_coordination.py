import os.path
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from mixed_vrep_env import *

ENV_NAME = 'Pioneer_p3dx'
#model_fn = 'dqn_coord_{}_weights.h5f'.format('pioneer_p3dx')

# Get the environment and extract the number of actions.
mix_env = MixedPioneerVrepEnv() #gym.make(ENV_NAME)
np.random.seed(123)
mix_env.seed(123)

#nb_actions = env.action_space.shape[0]

def get_dqn(model_fn, obs_dim, n_actions):
	if os.path.isfile(model_fn):
		model = Sequential()
		model.add(Flatten(input_shape=(1,) + obs_dim))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(n_actions))
		model.add(Activation('linear'))
		model.load_weights(model_fn)
		return model
	else:
		print('Model file not found')
		return None

dqn_follow_wall = get_dqn('follow_wall_dqn_pioneer_p3dx_weights.h5f',mix_env.observation_space_follow.shape,2)
#dqn_avoid_obstacle = get_dqn('',avoid_env) env.observation_space.shape
dqn_go_to_goal = get_dqn('gotogoal_dqn_pioneer_p3dx_weights.h5f',mix_env.observation_space.shape,len(mix_env.actions))

'''
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=32,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
'''

OBST_TRESHOLD = 0.3
WALL_TRESHOLD = 0.4

mix_env.reset()
for i in range(200):
	#Se houver alguma distancia muito pequena a saida mais baixa e ativada
	dist = mix_env.observation_follow
	action = 'goal'
	if min(dist) < OBST_TRESHOLD:
		action = 'avoid'
	elif min(dist[0], dist[1], dist[6], dist[7]) < WALL_TRESHOLD:
		action = 'follow'

	if action == 'follow':
		act = dqn_follow_wall.predict(mix_env.observation_follow.reshape([1,1,8]), batch_size=None, verbose=0, steps=1)
		mix_env.step_follow(act[0])
	else:
		act = dqn_go_to_goal.predict(mix_env.observation.reshape([1,1,3]), batch_size=None, verbose=0, steps=1)
		mix_env.step(np.argmax(act[0]))

#dqn.test(env, nb_episodes=2, visualize=True)

