# gym-afc

Open AI Gym Environment for Artificial Fovea Control on the visual object search task.

### Overview

The value-based model-free reinforcement learning for the artificial fovea control was
proposed and the computational simulation on the visual search task was conducted. A gaze
movement is the sequence of actions, and it is necessary to consider the temporal dependence of
the agent’s states in order to properly select the next action. We trained the agent to learn the
policy of the visual object search task in scene images of the MIT Scene Parsing Benchmark
Dataset[3] by applying the Actor-Critic algorithm[1][2].

#### Problem Settings

The standard reinforcement learning settings are used. For each time step, the agent observes
a visual input s and select an action a according to its policy function π(a|s). In return, the agent
gets the next state s’ from the transition function P(s’|s,a) and a reward r from the reward
function R(s,a). This process repeats until a terminal state.

#### Learning Algorithm
As a learning method, we adopt the Actor-Critic algorithm[1] which has both the value
function Q(s,a;w) and the policy function π(a|s;θ). At each time step, the parameter θ of the
policy function π(a|s;θ) is updated by using the state-action value Q(s,a;w) (Critic):

<div style="text-align: center;">
θ ← θ + η × Q(s,a;w)∇log π(a|s;θ)
</div>


and then the parameter w of the value function Q(s,a;w) is updated by using the TD error r + γ
Q(s’,a’;w) - Q(s,a;w) for the current action a (Actor):

<div style="text-align: center;">
w ← w + η × {r + γ Q(s’,a’;w) - Q(s,a;w)}∇Q(s,a;w)
</div>

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

```
```
