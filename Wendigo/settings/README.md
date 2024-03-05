***
# Settings Readme
***

### General Settings

###### Experiment Settings

**exp-name** (str): The name of this experiment

**seed** (int): Seed of the experiment

**torch-deterministic** (bool): If toggled, `torch.backends.cudnn.deterministic=False`

**cuda** (bool): If toggled, cuda will be enabled by default

**track** (bool): If toggled, this experiment will be tracked with Weights and Biases

**wandb-project-name** (str): The Weights and Biases' project name

**wandb-entity** (str): The entity (team) of Weights and Biases' project

**capture-video** (bool): Whether to capture videos of the agent performances

***

###### GraphQL Env Arguments

***env-id*** (str): The id of the environment

**schema-path** (str): the path for the stored schema of the application

***

###### Connection Settings

**appName** (str) : The name of the docker container ("dolevf/dvga")

**targetHost** (str): The hostname or ip of the target

**targetPort** (int): The port of the target

**targetPath** (str): The path of the GraohQL endpoint (default: `/graphql`)

**method** (str): The HTTP method of the GraphQL endpoint (default: `POST`)

**headers** (dict): The headers of the GraphQL endpoint (default: `{ content-type: application/json }`)

**useSSL** (bool): Whether to use SSL

**verifySSL** (bool): Whether to verify SSL 

**certPath** (str): The path to the certificate 

**keyPath** (str): The path to the key file 

**caPath** (str): The path to the CA file

***

### DoS Specific Settings

**max-depth** (int): The max depth of query recursion used in state/action.

**max-height** (int): The max repeat of a single location using a repeat method ex. field duplication or aliasing.

**multiplier** (int): The amount to increase/decrease per performing of an action

**reward-settings** (dict): The dictionary of rewards (can be +/-), there are the following:
- "repeat": reward given when the same query is attempted to be sent again, often the result of a invalid action.
- "new": reward given when a new query has been produced
- "rejected": reward given when query sent is rejected by the application
- "crash": reward given when the query sent causes the application to crash

**greedy** (bool): if current state should only be updated if we find a better delay query

***

### PPO Specific Settings

###### Environment arguments

***num-envs*** (int): The number of parallel game environments

***num-steps*** (int): The number of steps to run in each environment per policy rollout

***total-timesteps*** (int): Total timesteps of the experiments

***

###### Optimizer arguments

***learning-rate*** (float): The learning rate of the optimizer

***optimizer_eps*** (float): The epsilon for the optimizer

***anneal-lr*** (bool): Toggle learning rate annealing for policy and value networks

***

###### Agent arguments

***gamma*** (float): The discount factor gamma

***gae-lambda*** (float): The lambda for the general advantage estimation

***num-minibatches*** (int): The number of mini-batches

***update-epochs*** (int): The K epochs to update the policy

***norm-adv*** (bool): Toggles advantages normalization

***clip-coef*** (float): The surrogate clipping coefficient

***clip-vloss*** (bool): Toggles whether to use a clipped loss for the value function, as per the paper.

***ent-coef*** (float): Coefficient of the entropy

***vf-coef*** (float): Coefficient of the value function

***max-grad-norm*** (float): The maximum norm for the gradient clipping

***target-kl*** (float): The target KL divergence threshold

***

### DQN Specific Arguments

**save-model** (boolean): If the trained model should be saved

###### Environment arguments

***num-envs***: One environment as multi env is currently not supported

***total-timesteps*** (int): Total timesteps of the experiments

***

###### Optimizer arguments

***learning-rate*** (float): The learning rate of the optimizer

***

###### Agent arguments

**buffer-size** (int): The replay memory buffer size

**gamma** (float): The discount factor gamma

**tau** (float): The target network update rate

**target-network-frequency** (int): The timesteps it takes to update the target network

**batch-size** (int): The batch size of sample from the reply memory

**start-e** (float): The starting epsilon for exploration

**end-e** (float): The ending epsilon for exploration

**exploration-fraction** (float): The fraction of 'total-timesteps' it takes from start-e to go to end-e

**learning-starts** (int): Timesteps to start learning

**train-frequency** (int): The frequency of training

***