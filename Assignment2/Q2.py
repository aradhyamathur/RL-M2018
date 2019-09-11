#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### ITERATIVE METHOD

# In[29]:


GRID_DIM = 5
GRID_SIZE = GRID_DIM * GRID_DIM
A, A_ , B, B_ = np.array([0, 1]), np.array([4, 1]), np.array([0, 3]), np.array([2, 3])
LEFT, RIGHT, UP , DOWN = np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]),                         np.array([1, 0]) 
A_REWARD = +5.0
B_REWARD = +10.0
GAMMA = 0.9 
Pi_as = 0.25
ITERATIONS = 10000
ACTIONS = [LEFT, RIGHT , UP , DOWN]


# In[20]:


def get_reward(location):
    if location == A:
        return A_REWARD
    if location == B:
        return B_REWARD
    
    if location[0] >= GRID_DIM or location[1] < 0:
        return -1
    
    else:
        return 0
def get_state_reward(cur_state, action):
    if cur_state[0] == A[0] and cur_state[1] == A[1]:
        return A_, A_REWARD
    if cur_state[0] == B[0] and cur_state[1] == B[1]:
        return B_, B_REWARD
    
    n_loc = cur_state + action
    if n_loc[0] >= GRID_DIM or n_loc[0] < 0 or n_loc[1] >= GRID_DIM or n_loc[1] < 0:
        return cur_state, -1
    else:
        return n_loc, 0


# In[53]:


v_pi = np.zeros((GRID_DIM, GRID_DIM))
count = 0
for i in range(ITERATIONS):
    v_pi_ = np.zeros((GRID_DIM, GRID_DIM))
    for i in range(GRID_DIM):
        for j in range(GRID_DIM):
            for act in ACTIONS:
                cur_state = np.array([i, j])
                (n_i, n_j), reward = get_state_reward(cur_state, act)
                # equation 3.12
                v_pi_[i, j] += Pi_as * (reward + GAMMA * v_pi[n_i, n_j])
    count += 1
    if np.sum(np.abs(v_pi - v_pi_)) < 1e-4:
        break
    v_pi = v_pi_
                
                
    


# In[55]:


v_pi, count


# In[ ]:


############## method 2


# ### LINEAR ALGEBRA

# In[2]:


import numpy as np


# In[15]:


GRID_DIM = 5
GRID_SIZE = GRID_DIM * GRID_DIM
A, A_ , B, B_ = np.array([0, 1]), np.array([4, 1]), np.array([0, 3]), np.array([2, 3])
LEFT, RIGHT, UP , DOWN = np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]),                         np.array([1, 0]) 
A_REWARD = +10.0
B_REWARD = +5.0
GAMMA = 0.9 
Pi_as = 0.25
ITERATIONS = 10000
ACTIONS = [LEFT, RIGHT , UP , DOWN]


# In[4]:


def get_value(i, j):
    reward = 0.0
    for action in ACTIONS:
        cur_location = np.array([i, j]) 
        next_location = cur_location +  action
        reward += get_reward(cur_location, next_location)
    return Pi_as * reward 


# In[5]:


def get_reward(location, n_location):
    if location[0] == A[0] and location[1] == A[1]:
        return A_REWARD
    if location[0] == B[0] and location[1] == B[1]:
        return B_REWARD
    
    if n_location[0] >= GRID_DIM or n_location[1] < 0:
        return -1
    else:
        return 0


# In[16]:


b  = []
for i in range(GRID_DIM):
    for j in range(GRID_DIM):
        
        b.append(get_value(i, j))
b =  np.array(b)
print(b.shape)
b


# In[17]:


def get_state_reward(cur_state, action):
    if cur_state[0] == A[0] and cur_state[1] == A[1]:
        return A_, A_REWARD
    if cur_state[0] == B[0] and cur_state[1] == B[1]:
        return B_, B_REWARD
    
    n_loc = cur_state + action
    if n_loc[0] >= GRID_DIM or n_loc[0] < 0 or n_loc[1] >= GRID_DIM or n_loc[1] < 0:
        return cur_state, -1
    else:
        return n_loc, 0


# In[42]:


ACTIONS = [LEFT, RIGHT , UP , DOWN]
b = np.zeros(25)


# In[43]:


a_mat = np.zeros((GRID_DIM*GRID_DIM, GRID_DIM*GRID_DIM))
GAMMA = 0.9
for i in range(GRID_DIM):
    
    for  j in range(GRID_DIM):
        
        for index, act in enumerate(ACTIONS):
                cur_loc  = np.array([i,j])
                cur_index = i*5 + j
                next_loc, reward = get_state_reward(cur_loc, act)
                index_ = next_loc[0] * 5 + next_loc[1]
                b[cur_index] += reward * Pi_as
                a_mat[cur_index, index_] += (-GAMMA *  Pi_as)
                        # general case
for i in range(GRID_DIM*GRID_DIM):
    for j in range(GRID_DIM*GRID_DIM):
        if i == j:
            a_mat[i,j] += 1.
#         else:
#             a_mat[i, j] = -GAMMA * Pi_as


# In[44]:


b


# In[45]:


y = np.linalg.solve(a_mat, b)
print(np.round(y, decimals=1).reshape(5, 5))


# In[ ]:





# In[ ]:




