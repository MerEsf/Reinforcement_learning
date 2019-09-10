# ## Packages.
# 1. jdc: Jupyter magic that allows defining classes over multiple jupyter notebook cells.
# 2. numpy: the fundamental package for scientific computing with Python.
# 3. matplotlib: the library for plotting graphs in Python.
# 4. RL-Glue: the library for reinforcement learning experiments.
# 5. BaseEnvironment, BaseAgent: the base classes from which we will inherit when creating the environment and agent classes in order for them to support the RL-Glue framework.
# 6. operator.add: the function that is useful adding tuples.
# 7. Manager: the file allowing for visualization and testing.
# 8. itertools.product: the function that can be used easily to compute permutations.
# 9. tqdm.tqdm: Provides progress bars for visualizing the status of loops.

import jdc
# --
import numpy as np
# --
from operator import add
# --
from rl_glue import RLGlue
# --
from Agent import BaseAgent 
from Environment import BaseEnvironment  
# --
from manager import Manager
# --
from itertools import product
# --
from tqdm import tqdm

# Create empty CliffWalkEnvironment class.
# These methods will be filled in later cells.
class CliffWalkEnvironment(BaseAgent):
    def env_init(self, agent_info={}):
        raise NotImplementedError

    def env_start(self, state):
        raise NotImplementedError

    def env_step(self, reward, state):
        raise NotImplementedError

    def env_end(self, reward):
        raise NotImplementedError
        
    def env_cleanup(self, reward):
        raise NotImplementedError
    
    # helper method
    def state(self, loc):
        raise NotImplementedError


# ## env_init()
# 
# The first function we add to the environment is the initialization function which is called once when an environment object is created. In this function, the grid dimensions and special locations (start and goal locations and the cliff locations) are stored for easy use later.

get_ipython().run_cell_magic   

# ## *Implement* state()

get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '#GRADED FUNCTION: [state]\n\n# is function to return a correct single index as \n# the state (see the logic for this in the previous cell.)\n# Lines: 1\ndef state(self, loc):\n    k = 12 * loc[0] + loc[1]\n    return k\n    ### END CODE HERE ###')


def test_state():
    env = CliffWalkEnvironment()
    env.env_init({"grid_height": 4, "grid_width": 12})
    coords_to_test = [(0, 0), (0, 11), (1, 5), (3, 0), (3, 9), (3, 11)]
    true_states = [0, 11, 17, 36, 45, 47]
    output_states = [env.state(coords) for coords in coords_to_test]
    assert(output_states == true_states)
test_state()


# ## env_start()
# 
# In env_start(), we initialize the agent location to be the start location and return the state corresponding to it as the first state for the agent to act upon. Additionally, we also set the reward and termination terms to be 0 and False respectively as they are consistent with the notion that there is no reward nor termination before the first action is even taken.

# In[7]:


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# Do not modify this cell!\n\n# Work Required: No.\ndef env_start(self):\n    """The first method called when the episode starts, called before the\n    agent starts.\n\n    Returns:\n        The first state from the environment.\n    """\n    reward = 0\n    # agent_loc will hold the current location of the agent\n    self.agent_loc = self.start_loc\n    # state is the one dimensional state representation of the agent location.\n    state = self.state(self.agent_loc)\n    termination = False\n    self.reward_state_term = (reward, state, termination)\n\n    return self.reward_state_term[1]')


# ## *Implement* env_step()
# 
# In the Cliff Walking environment, agents move around using a 4-cell neighborhood called the Von Neumann neighborhood (https://en.wikipedia.org/wiki/Von_Neumann_neighborhood). Thus, the agent has 4 available actions at each state. Three of the actions have been implemented for you and your first task is to implement the logic for the fourth action (Action UP).


get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '#GRADED FUNCTION: [env_step]\n\n# Work Required: Yes. Fill in the code for action UP and implement the logic for reward and termination.\n# Lines: ~7.\ndef env_step(self, action):\n    """A step taken by the environment.\n\n    Args:\n        action: The action taken by the agent\n\n    Returns:\n        (float, state, Boolean): a tuple of the reward, state,\n            and boolean indicating if it\'s terminal.\n    """\n\n    if action == 0: # UP (Task 1)\n        possible_next_loc = tuple(map(add, self.agent_loc, (-1, 0)))\n        if possible_next_loc[0] >= 0: # Within Bounds?\n            self.agent_loc = possible_next_loc\n        else:\n        # Hint: Look at the code given for the other actions and think about the logic in them.\n            pass # Stay \n        ### END CODE HERE ###\n    elif action == 1: # LEFT\n        possible_next_loc = tuple(map(add, self.agent_loc, (0, -1)))\n        if possible_next_loc[1] >= 0: # Within Bounds?\n            self.agent_loc = possible_next_loc\n        else:\n            pass # Stay.\n    elif action == 2: # DOWN\n        possible_next_loc = tuple(map(add, self.agent_loc, (+1, 0)))\n        if possible_next_loc[0] < self.grid_h: # Within Bounds?\n            self.agent_loc = possible_next_loc\n        else:\n            pass # Stay.\n    elif action == 3: # RIGHT\n        possible_next_loc = tuple(map(add, self.agent_loc, (0, +1)))\n        if possible_next_loc[1] < self.grid_w: # Within Bounds?\n            self.agent_loc = possible_next_loc\n        else:\n            pass # Stay.\n    else: \n        raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")\n\n    reward = -1\n    terminal = False\n\n    ### START CODE HERE ###\n    # Hint: Consider the initialization of reward and terminal variables above. Then, note the \n    # conditional statements and comments given below and carefully ensure to set the variables reward \n    # and terminal correctly for each case.\n    if self.agent_loc == self.goal_loc: # Reached Goal!\n        terminal = True\n        pass\n    elif self.agent_loc in self.cliff: # Fell into the cliff!\n        reward = -100\n        self.agent_loc = self.start_loc\n       # terminal = False\n        pass\n    else: \n        #reward = -1\n        #terminal = False\n        pass\n    ### END CODE HERE ###\n    \n    self.reward_state_term = (reward, self.state(self.agent_loc), terminal)\n    return self.reward_state_term')


# In[9]:


def test_action_up():
    env = CliffWalkEnvironment()
    env.env_init({"grid_height": 4, "grid_width": 12})
    env.agent_loc = (0, 0)
    env.env_step(0)
    assert(env.agent_loc == (0, 0))
    
    env.agent_loc = (1, 0)
    env.env_step(0)
    assert(env.agent_loc == (0, 0))
test_action_up()

def test_reward():
    env = CliffWalkEnvironment()
    env.env_init({"grid_height": 4, "grid_width": 12})
    env.agent_loc = (0, 0)
    reward_state_term = env.env_step(0)
    assert(reward_state_term[0] == -1 and reward_state_term[1] == env.state((0, 0)) and
           reward_state_term[2] == False)
    
    env.agent_loc = (3, 1)
    reward_state_term = env.env_step(2)
    assert(reward_state_term[0] == -100 and reward_state_term[1] == env.state((3, 0)) and
           reward_state_term[2] == False)
    
    env.agent_loc = (2, 11)
    reward_state_term = env.env_step(2)
    assert(reward_state_term[0] == -1 and reward_state_term[1] == env.state((3, 11)) and
           reward_state_term[2] == True)
test_reward()

get_ipython().run_cell_magic('add_to', 'CliffWalkEnvironment', '\n# Do not modify this cell!\n\n# Work Required: No.\ndef env_cleanup(self):\n    """Cleanup done after the environment ends"""\n    self.agent_loc = self.start_loc')


# Create empty TDAgent class.


class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        raise NotImplementedError
        
    def agent_start(self, state):
        raise NotImplementedError

    def agent_step(self, reward, state):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError

    def agent_cleanup(self):        
        raise NotImplementedError
        
    def agent_message(self, message):
        raise NotImplementedError




# # agent_start()
# 
# In agent_start(), we choose an action based on the initial state and policy we are evaluating. We also cache the state so that we can later update its value when we perform a Temporal Difference update. Finally, we return the action chosen so that the RL loop can continue and the environment can execute this action.

# ## *Implement* agent_step()

get_ipython().run_cell_magic('add_to', 'TDAgent', '\n#[GRADED] FUNCTION: [agent_step]\n\n# Work Required: Yes. Fill in the TD-target and update.\n# Lines: ~2.\ndef agent_step(self, reward, state):\n    """A step taken by the agent.\n    Args:\n        reward (float): the reward received for taking the last action taken\n        state (Numpy array): the state from the\n            environment\'s step after the last action, i.e., where the agent ended up after the\n            last action\n    Returns:\n        The action the agent is taking.\n    """\n    ### START CODE HERE ###\n    # Hint: We should perform an update with the last state given that we now have the reward and\n    # next state. We break this into two steps. Recall for example that the Monte-Carlo update \n    # had the form: V[S_t] = V[S_t] + alpha * (target - V[S_t]), where the target was the return, G_t.\n    target = reward + self.discount * self.values[state]\n    self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])\n    ### End CODE HERE ###\n    \n    # Having updated the value for the last state, we now act based on the current \n    # state, and set the last state to be current one as we will next be making an \n    # update with it when agent_step is called next once the action we return from this function \n    # is executed in the environment.\n\n    action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])\n    self.last_state = state\n\n    return action')

get_ipython().run_cell_magic('add_to', 'TDAgent', '\n#[GRADED] FUNCTION: [agent_end]\n\n# Work Required: Yes. Fill in the TD-target and update.\n# Lines: ~2.\ndef agent_end(self, reward):\n    """Run when the agent terminates.\n    Args:\n        reward (float): the reward the agent received for entering the terminal state.\n    """\n\n    ### START CODE HERE ###\n    # Hint: Here too, we should perform an update with the last state given that we now have the \n    # reward. Note that in this case, the action led to termination. Once more, we break this into \n    # two steps, computing the target and the update itself that uses the target and the \n    # current value estimate for the state whose value we are updating.\n    target = reward\n    self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])\n    ### END CODE HERE ###')

get_ipython().run_cell_magic('add_to', 'TDAgent', '\n# Do not modify this cell!\n\n# Work Required: No.\ndef agent_cleanup(self):\n    """Cleanup done after the agent ends."""\n    self.last_state = None')


def test_td_updates():

    agent = TDAgent()
    policy_list = np.array([[1.], [1.]])
    agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
    agent.values = np.array([0., 1.])
    agent.agent_start(0)
    reward = -1
    next_state = 1
    agent.agent_step(reward, next_state)
    assert(np.isclose(agent.values[0], -0.001) and np.isclose(agent.values[1], 1.))

    agent = TDAgent()
    policy_list = np.array([[1.]])
    agent.agent_init({"policy": np.array(policy_list), "discount": 0.99, "step_size": 0.1})
    agent.values = np.array([0.])
    agent.agent_start(0)
    reward = -100
    next_state = 0
    agent.agent_end(reward)
    assert(np.isclose(agent.values[0], -10))
    
test_td_updates()


get_ipython().run_line_magic('matplotlib', 'notebook')


def run_experiment(env_info, agent_info, 
                   num_episodes=5000,
                   experiment_name=None,
                   plot_freq=100,
                   true_values_file=None,
                   value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)
    for episode in range(1, num_episodes + 1):
        rl_glue.rl_episode(0) # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    if true_values_file is not None:
       
        manager.run_tests(values, value_error_threshold)
    
    return values

env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}

# The Optimal Policy that strides just along the cliff
policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [1, 0, 0, 0]
for i in range(24, 35):
    policy[i] = [0, 0, 0, 1]
policy[35] = [0, 0, 1, 0]

agent_info.update({"policy": policy})

true_values_file = "optimal_policy_value_fn.npy"
_ = run_experiment(env_info, agent_info, num_episodes=5000, experiment_name="Policy Evaluation on Optimal Policy",
                   plot_freq=500, true_values_file=true_values_file)

policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25

w = env_info['grid_width']
h = env_info['grid_height']

for i in range (w , (h - 1)* w + 1):
    policy[i] = [1, 0, 0, 0]

for k in range (0, h-1):
    policy[(k+1)*w -1] = [0, 0, 1, 0]

pass

### AUTO-GRADER TESTS FOR POLICY EVALUATION WITH SAFE POLICY
agent_info.update({"policy": policy})
v = run_experiment(env_info, agent_info,
               experiment_name="Policy Evaluation On Safe Policy",
               num_episodes=5000, plot_freq=500)

env_info = {"grid_height": 4, "grid_width": 12}
agent_info = {"discount": 1, "step_size": 0.01}

policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [0.9, 0.1/3., 0.1/3., 0.1/3.]
for i in range(24, 35):
    policy[i] = [0.1/3., 0.1/3., 0.1/3., 0.9]
policy[35] = [0.1/3., 0.1/3., 0.9, 0.1/3.]
agent_info.update({"policy": policy})
agent_info.update({"step_size": 0.01})


arr = []
from tqdm import tqdm
for i in tqdm(range(30)):
    env_info['seed'] = i
    agent_info['seed'] = i
    v = run_experiment(env_info, agent_info,
                   experiment_name="Policy Evaluation On Optimal Policy",
                   num_episodes=5000, plot_freq=10000)
    arr.append(v)
average_v = np.array(arr).mean(axis=0)


