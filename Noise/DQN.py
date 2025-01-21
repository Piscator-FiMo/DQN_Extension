"""Now, let's define our model. But first, let's quickly recap what a DQN
is.


DQN algorithm
=============

Our environment is deterministic, so all equations presented here are
also formulated deterministically for the sake of simplicity. In the
reinforcement learning literature, they would also contain expectations
over stochastic transitions in the environment.

Our aim will be to train a policy that tries to maximize the discounted,
cumulative reward
$R_{t_0} = sum_{t=t_0}^{infty} gamma^{t - t_0} r_t$, where $R_{t_0}$
is also known as the *return*. The discount, $gamma$, should be a
constant between $0$ and $1$ that ensures the sum converges. A lower
$gamma$ makes rewards from the uncertain far future less important for
our agent than the ones in the near future that it can be fairly
confident about. It also encourages agents to collect reward closer in
time than equivalent rewards that are temporally far away in the future.

The main idea behind Q-learning is that if we had a function
$Q^*: State times Action rightarrow mathbb{R}$, that could tell us
what our return would be, if we were to take an action in a given state,
then we could easily construct a policy that maximizes our rewards:

$$pi^*(s) = arg!max_a  Q^*(s, a)$$

However, we don't know everything about the world, so we don't have
access to $Q^*$. But, since neural networks are universal function
approximators, we can simply create one and train it to resemble $Q^*$.

For our training update rule, we'll use a fact that every $Q$ function
for some policy obeys the Bellman equation:

$$Q^{pi}(s, a) = r + gamma Q^{pi}(s', pi(s'))$$

The difference between the two sides of the equality is known as the
temporal difference error, $delta$:

$$delta = Q(s, a) - (r + gamma max_a' Q(s', a))$$

To minimize this error, we will use the [Huber
loss](https://en.wikipedia.org/wiki/Huber_loss). The Huber loss acts
like the mean squared error when the error is small, but like the mean
absolute error when the error is large - this makes it more robust to
outliers when the estimates of $Q$ are very noisy. We calculate this
over a batch of transitions, $B$, sampled from the replay memory:

$$mathcal{L} = frac{1}{|B|}sum_{(s, a, s', r)  in  B} mmathcal{L}(delta)$$

$$begin{aligned}
text{where} quad mathcal{L}(delta) = begin{cases}
  frac{1}{2}{delta^2}  & text{for } |delta| le 1,
  |delta| - frac{1}{2} & text{otherwise.}
end{cases}
end{aligned}$$

Q-network
---------

Our model will be a feed forward neural network that takes in the
difference between the current and previous screen patches. It has two
outputs, representing $Q(s, mathrm{left})$ and $Q(s, mathrm{right})$
(where $s$ is the input to the network). In effect, the network is
trying to predict the *expected return* of taking each action given the
current input.

"""


from libraries import *

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 512)
        self.layer3 = NoisyLinear(512, n_actions, sigma_init=0.3)

        self.layerNoise = NoisyLinear(n_observations, 128, sigma_init=0.1)
        self.layer4 = nn.Linear(512, n_actions)
        self.layer5 = nn.Linear(128, 128)

        self.fc_adv = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # Without Noise
    def forward1(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer4(x)

    # Noisy
    def forward2(self, x):
        x = F.relu(self.layerNoise(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer4(x))
        return x

    # Dueling
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer5(x))

        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()