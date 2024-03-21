***
# Wendigo
## Deep Reinforcement Learning for Denial-of-Service Query Discovery in GraphQL

This repository is the official release of the code used for the Wendigo Paper published in Deep Learning Security and Privacy Workshop (DSLP) 2024, co-located with IEEE S&P.

If you plan to use this repository in your projects, please cite the following paper:

```bibtex
@inproceedings{mcfadden2024wendigo,
  title = {Wendigo: Deep Reinforcement Learning for Denial-of-Service Query Discovery in GraphQL},
  author = {McFadden, Shae and Maugeri, Marcello and Hicks, Chris and Mavroudis, Vasilis and Pierazzi, Fabio},
  booktitle = {Proc. of the {IEEE} Workshop on Deep Learning Security and Privacy ({DLSP})},
  year = {2024},
}
```

**Disclaimer**: Please note that the code in this repository is only a research prototype and may generate damaging queries. Do not use against real systems without prior, written consent of the targets. This code is released under a "Modified (Non-Commercial) BSD License" (see the terms [here](./LICENSE)).

***

### Use Instructions
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip3 install -r requirements.txt`
4. `pip3 install .`
5. `cd Wendigo`
6. `python3 main.py` 

***

### Wendigo Code Breakdown
The Wendigo code consists of the following sections:
- **attack**: The code and results used to run the DoS impact evaluation which utilized the found query to perform a mock DoS attack.
- **environments**: The environment is broken down into a parent and subclass.
  - _GraphQLEnv.py_: handles the docker and connection,
  - _GraphQLDoSEnv.py_: performs handles the mappings between agent's state/action and queries.
- **models**: This directory contains the code for PPO and Random.
  - _PPO.py_: Contains the PPO code utilized for the evaluation of the paper and is a modified version of cleanrl's PPO implementation.
  - _Random.py_: Contains the implementation used for random evaluation of the paper (note: greedy is a setting in the attack settings).
- **results**: This directory contains pickles of the results presented in the paper.
- **schemas**: This contains the DVGA schema used to inform the environment for state to query space mappings.
- **settings**: This directory contains the settings files used in the evaluation (Has its own read me for clarity on the settings).
- **utils**: Contains helper functions and functions to interface between the agent and environment.
- **main.py**: This file is used to run the application by specifying the variables below the application will load the appropriate setting file and run the experiment.
  - _MODEL_: 'PPO' or 'Random'
  - _TEST_: currently only 'DoS' is supported
  - _DESC_: 'Regular', 'Large', 'Greedy-Regular' or 'Greedy-Large'

***
