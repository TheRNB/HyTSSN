import pymonntorch as pymo

class LIF (pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=None, required=True)
        self.u_reset = self.parameter("u_reset", default=None, required=True)
        neural_group.threshold = self.parameter("threshold", default=None, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.refactory_period = self.parameter("refractory", default=False, required=False)
        self.k = self.parameter("kwta", default=None, required=False)
        if self.k is not None and self.k > neural_group.size:
            raise Exception("k value in KWinnerTakesAll can't be more than NG.size.")

        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (neural_group.threshold - self.u_reset) #returns the length of values LIF can get
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= neural_group.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = -(neural_group.u - self.u_rest)
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest 
            input_u[neural_group.is_refactory] = 0
        neural_group.u += (leakage + input_u) / self.tau * neural_group.network.dt

        if self.k is not None:
            kwta(self, neural_group)

        neural_group.spike = neural_group.u >= neural_group.threshold
        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return
    

class ELIF (pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=None, required=True)
        self.u_reset = self.parameter("u_reset", default=None, required=True)
        neural_group.threshold = self.parameter("threshold", default=None, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.sharpness = self.parameter("sharpness", default=None, required=True)
        self.firing_threshold = self.parameter("phi", default=None, required=True)
        self.refactory_period = self.parameter("refractory", default=False, required=False)

        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (neural_group.threshold - self.u_reset) #returns the length of values LIF can get
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= neural_group.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = - (neural_group.u - self.u_rest) + (self.sharpness * pymo.torch.exp((neural_group.u - self.firing_threshold)/self.sharpness))
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest
            input_u[neural_group.is_refactory] = 0
        neural_group.u += (leakage + input_u) / self.tau * neural_group.network.dt

        neural_group.spike = neural_group.u >= neural_group.threshold
        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return


class AELIF (pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=None, required=True)
        self.u_reset = self.parameter("u_reset", default=None, required=True)
        neural_group.threshold = self.parameter("threshold", default=None, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.sharpness = self.parameter("sharpness", default=None, required=True)
        self.firing_threshold = self.parameter("phi", default=None, required=True)
        self.A_param = self.parameter("A", default=None, required=True)
        self.B_param = self.parameter("B", default=None, required=True)
        self.tau_w = self.parameter("tau_w", default=None, required=True)
        self.refactory_period = self.parameter("refractory", default=False, required=False)

        neural_group.adaptation = neural_group.vector(mode="zeros")
        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (neural_group.threshold - self.u_reset) #returns the length of values LIF can get
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= neural_group.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = - (neural_group.u - self.u_rest) + (self.sharpness * pymo.torch.exp((neural_group.u - self.firing_threshold)/self.sharpness)) - (self.R * neural_group.adaptation)
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest
            input_u[neural_group.is_refactory] = 0
        neural_group.u += ((leakage + input_u) / self.tau) * neural_group.network.dt

        neural_group.spike = neural_group.u >= neural_group.threshold

        memory = (self.A_param * (neural_group.u - self.u_rest) - neural_group.adaptation) / self.tau_w
        effect = self.B_param
        neural_group.adaptation += (memory) * neural_group.network.dt
        neural_group.adaptation[neural_group.spike] += (effect) * neural_group.network.dt

        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return

def kwta(self, neural_group):
    will_spike = neural_group.u >= neural_group.threshold
    spike_indices = pymo.torch.nonzero(will_spike).squeeze()
    if spike_indices.numel() > self.k:
        sorted_indices = sorted(spike_indices.tolist(), key=lambda x: -neural_group.u[x])
        neural_group.u[sorted_indices[self.k:]] = self.u_reset
    return

class acticityHomeostasis(pymo.Behavior):
    def initialize(self, neural_group):
        self.window_size = self.parameter("window_size", default=None, required=True)
        self.activity_rate = self.parameter("activity_rate", default=None, required=True)
        self.updating_rate = self.parameter("updating_rate", default=None, required=True)
        self.decay_rate = self.parameter("decay_rate", default=1.00, required=False)
        neural_group.threshold = pymo.torch.tensor([neural_group.threshold for _ in range(neural_group.size)])
        self.current_reward_total = pymo.np.zeros(shape=neural_group.threshold.shape)

        self.reward = 1.00
        self.punishment = -1.00 * self.activity_rate / (self.window_size - self.activity_rate)
        #print("here with ", self.reward, self.punishment)
        return super().initialize(neural_group)

    def forward(self, neural_group):
        for i in range(len(neural_group.spike)):
            if neural_group.spike[i] == True:
                self.current_reward_total[i] += self.reward
            else:
                self.current_reward_total[i] += self.punishment
        if neural_group.iteration % self.window_size == 0:
            change = -self.current_reward_total * self.updating_rate
            neural_group.threshold -= change
            self.current_reward_total = pymo.np.zeros(shape=neural_group.threshold.shape)
            self.updating_rate *= self.decay_rate
            #print("changed weights with ", neural_group.threshold)
        return super().forward(neural_group)

class voltageHomeostasis(pymo.Behavior):
    def initialize(self, neural_group):
        self.activity_rate = self.parameter("activity_rate", default=None, required=True)
        self.max_tao = self.parameter("max_tao", default=self.activity_rate, required=False)
        self.min_tao = self.parameter("min_tao", default=self.activity_rate, required=False)
        self.eta = self.parameter("eta", default=None, required=True)
        
        self.current_reward_total = pymo.np.zeros(shape=neural_group.spike.shape)
        return super().initialize(neural_group)

    def forward(self, neural_group):
        greater = (neural_group.u > self.max_tao) * (neural_group.u - self.max_tao)
        smaller = (neural_group.u < self.min_tao) * (neural_group.u - self.min_tao)

        change = (greater + smaller) * self.eta

        for i in range(len(self.current_reward_total)):
            self.current_reward_total[i] += change[i]
        neural_group.u -= self.current_reward_total
        return super().forward(neural_group)