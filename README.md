# gym-afc

Open AI Gym Environment for Artificial Fovea Control on the visual object search task.

### Overview

The value-based model-free reinforcement learning for the artificial fovea control was
proposed and the computational simulation on the visual search task was conducted. A gaze
movement is the sequence of actions, and it is necessary to consider the temporal dependence of
the agent’s states in order to properly select the next action. We trained the agent to learn the
policy of the visual object search task in scene images of the MIT Scene Parsing Benchmark
Dataset[3] by applying the Actor-Critic algorithm[1][2].

### Methodology

#### Problem Settings

The standard reinforcement learning settings are used. For each time step, the agent observes
a visual input s and select an action a according to its policy function π(a|s). In return, the agent
gets the next state s’ from the transition function P(s’|s,a) and a reward r from the reward
function R(s,a). This process repeats until a terminal state.

#### Learning Algorithm
As a learning method, we adopt the Actor-Critic algorithm[1] which has both the value
function Q(s,a;w) and the policy function π(a|s;θ). At each time step, the parameter θ of the
policy function π(a|s;θ) is updated by using the state-action value Q(s,a;w) (Critic):

<p align="center">
θ ← θ + η × Q(s,a;w)∇log π(a|s;θ)
</p>

and then the parameter w of the value function Q(s,a;w) is updated by using the TD error r + γ
Q(s’,a’;w) - Q(s,a;w) for the current action a (Actor):

<p align="center">
w ← w + η × {r + γ Q(s’,a’;w) - Q(s,a;w)}∇Q(s,a;w)
</p>

where η is a learning rate and γ is a discount rate. The Actor-Critic algorithm is more valid and
efficient for the general situation than the Q-learning because the heuristic action selections
such as softmax and ε-gleedy in Q-learning do not acquire the direct mapping between the
action and the state.

#### Scene Image Dataset

The MIT Scene Parsing Benchmark Dataset was selected as the material for the visual object
search task. It is a common large image dataset for analyzing and modeling saliency and spatial
attention. When sampling the new state s’ from the transition function P(s’|s,a), the image
processing was performed to represent the human vision, where the resolution drops from the
fovea to the periphery. Each image has a corresponding semantic segmentation image, which is
used to implement the reward function R(s,a).

#### Foveated Imaging

As an observation of the agent, we use [Image Foveation Python](https://github.com/ouyangzhibo/Image_Foveation_Python) for the foveated image processing.
The function `foveat_img()` is the implemention according to the Salicon method[4][5].

### Usage

**Random walking on the visual search**

```python
import numpy as np
from environment import AfcEnv

env = AfcEnv()
env.reset()
observation = env.reset()
for _ in range(10000):
    env.render(mode="rgb_array")
    action = env.action_space.sample()  # action sampled at random
    observation, reward, done, _ = env.step(action)

    if done:
        env.reset()
```

### References

- [1] Vijaymohan Konda and John N. Tsitsiklis. 2002. Actor-critic algorithms. Ph.D. Dissertation. Massachusetts Institute of Technology, USA. Order Number: AAI0804543.
- [2] Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. & Kavukcuoglu, K.. (2016). Asynchronous Methods for Deep Reinforcement Learning. Proceedings of The 33rd International Conference on Machine Learning, in Proceedings of Machine Learning Research 48:1928-1937 Available from http://proceedings.mlr.press/v48/mniha16.html .
- [3] Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017.
- [4] Perry, Jeffrey S., and Wilson S. Geisler. "Gaze-contingent real-time simulation of arbitrary visual fields." Human vision and electronic imaging VII. Vol. 4662. International Society for Optics and Photonics, 2002.
- [5] Jiang, Ming, et al. "Salicon: Saliency in context." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
