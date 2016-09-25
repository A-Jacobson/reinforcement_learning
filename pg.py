
# kaparthy policy gradients with keras as a backend
import numpy as np
import cPickle as pickle
import gym
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, TimeDistributed, Convolution2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


H = 200 # number of hidden layer neurons
batch_size = 10 # how many episodes to do a param update
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False
train_eps = 10
D = [1, 1, 80, 80]

# env settings
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs, dlogps, drs = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
nframes = 1


def karpathy_simple_pgnet(env, dropout=0.5, learning_rate=1e-4, **args):
    S = Input(shape=[agent.input_dim])
    h = Dense(200, activation='relu', init='he_normal')(S)
    h = Dropout(dropout)(h)
    V = Dense(env.action_space.n, activation='sigmoid',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate) )
    return model


# if resume:
#     model = karpathy_simple_pgnet(D)
#     model.load_weights('test.h5')
#
# else:
#     model = karpathy_simple_pgnet(D)

model = pgconvnet(env, D)

def prepro(I):
    """
    preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by a factor of 2
    I[I == 144] = 0 # erase background (bg type 1)
    I[I == 109] = 0 # erase background (bg type 2)
    I[I != 0] = 1 # set everything else (ball, paddles) to 1
    return I.astype(np.float)

def discount_rewards(r):
    """
    Take 1D float of array of rewards and compute discounted reward
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs, dlogps, drs = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
nframes = 1

model = karpathy_simple_pgnet(D)

while True:
    if render: env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob = np.squeeze(model.predict(x.reshape(1, D))).flatten()
    action = np.random.choice( env.action_space.n, 1, p=aprob/np.sum(aprob) )[0]
    xs.append(x) # keep track of all our states

    y = np.zeros([env.action_space.n])
    y[action] = 1

    dlogps.append(y) # grad that encourages the action that was tak


    # step env and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += float(reward)

    drs.append(float(reward))
    # record reward

    if done:
        episode_number += 1


        epx = np.vstack(xs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        xs, dlogps, drs = [], [], []

        # compute discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)


        epdlogp *= discounted_epr


        if episode_number % batch_size == 0:
            print epx.shape, epdlogp.shape
            model.fit(epx, epdlogp, nb_epoch=1, verbose=2)

        # book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 100 == 0: model.save('test.h5')
        reward_sum = 0
        observation = env.reset()
        prev_x = None


    if reward != 0: # pong has either +1 or -1 reward exactly when game ends
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
