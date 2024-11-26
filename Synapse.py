import pymonntorch as pymo


class all_to_all_connection(pymo.Behavior):
    # TODO: migrate to PyTorch
    def initialize(self, synapse):
        synapse.excitatory = self.parameter(
            "pre_is_excitate", default=True, required=False
        )
        synapse.JZero = self.parameter("J0", default=1, required=False)
        synapse.std = self.parameter("std", default=0.01, required=False)

        number_of_source_neurons = synapse.src.size
        normal_dist = "normal({0}, {1})".format(
            synapse.JZero / number_of_source_neurons,
            synapse.std / number_of_source_neurons,
        )

        synapse.W = synapse.matrix(normal_dist)
        synapse.I = synapse.dst.vector()
        return

    def forward(self, synapse):
        if synapse.excitatory:
            # synapse.I = pymo.torch.sum(synapse.W[synapse.src.spike], axis=0) #TODO DOES NOT SUM CORRECTlY
            synapse.I = pymo.np.zeros(synapse.W.shape[1])
            for i in range(synapse.src.size):
                if synapse.src.spike[i]:
                    for j in range(synapse.dst.size):
                        synapse.I[j] += synapse.W[i, j]
            # print(synapse.I)
            # print(synapse.src.spike, synapse.W)
        else:
            synapse.I = pymo.torch.sum(
                pymo.torch.zeros(synapse.W[synapse.src.spike].shape)
                - synapse.W[synapse.src.spike],
                axis=0,
            )
        return


class random_fixed_prob_connection(pymo.Behavior):
    # TODO: migrate to PyTorch
    def initialize(self, synapse):
        synapse.excitatory = self.parameter(
            "pre_is_excitate", default=True, required=False
        )
        synapse.JZero = self.parameter("J0", default=1, required=False)
        synapse.probability = self.parameter("prob", default=0.1, required=False)
        synapse.std = self.parameter("std", default=0.01, required=False)
        number_of_source_neurons = synapse.src.size

        synapse.W = synapse.matrix("uniform")
        synapse.W = pymo.torch.where(
            synapse.W < synapse.probability,
            pymo.torch.normal(
                mean=(synapse.JZero / (synapse.probability * number_of_source_neurons)),
                std=synapse.std / (synapse.probability * number_of_source_neurons),
                size=synapse.W.shape,
            ),
            pymo.torch.tensor(0.0),
        )
        synapse.I = synapse.dst.vector()
        return

    def forward(self, synapse):
        if synapse.excitatory:
            synapse.I = pymo.torch.sum(synapse.W[synapse.src.spike], axis=0)
        else:
            synapse.I = pymo.torch.sum(
                pymo.torch.zeros(synapse.W[synapse.src.spike].shape)
                - synapse.W[synapse.src.spike],
                axis=0,
            )
        return


class random_fixed_edgeCount_connection(pymo.Behavior):
    # TODO: migrate to PyTorch
    def initialize(self, synapse):
        synapse.excitatory = self.parameter(
            "pre_is_excitate", default=True, required=False
        )
        synapse.JZero = self.parameter("J0", default=1, required=False)
        synapse.C_const = self.parameter("C", default=0, required=True)
        synapse.std = self.parameter("std", default=0.01, required=False)

        synapse.W = synapse.matrix("zeros")
        for j in range(synapse.W.shape[1]):
            indices = pymo.torch.randperm(synapse.W.shape[0])[
                : synapse.C_const
            ]  # Randomly permute indices and select the first C
            for index in indices:
                synapse.W[index, j] = pymo.torch.normal(
                    mean=(synapse.JZero / synapse.C_const),
                    std=(synapse.std / synapse.C_const),
                    size=(1, 1),
                )
        synapse.I = synapse.dst.vector()
        return

    def forward(self, synapse):
        if synapse.excitatory:
            synapse.I = pymo.torch.sum(synapse.W[synapse.src.spike], axis=0)
        else:
            synapse.I = pymo.torch.sum(
                pymo.torch.zeros(synapse.W[synapse.src.spike].shape)
                - synapse.W[synapse.src.spike],
                axis=0,
            )
        return


class Learning(pymo.Behavior):
    # TODO: migrate to PyTorch
    def initialize(self, synapse):
        self.learningFunction = self.parameter("function", default=None, required=True)
        synapse.reward = self.parameter("reward", default=None, required=True)
        synapse.time_window = self.parameter("time", default=0, required=True)
        self.procedure = self.parameter("procedure", default=None, required=True)
        self.curr_time = 0
        self.min_weight = 0.1
        self.max_weight = 10

        if self.learningFunction == "flat":
            self.learningFunction = Learning.flatSTDP_delta_function
        else:
            self.learningFunction = Learning.STDP_delta_function

        if self.procedure == "rstdp":
            self.procedure = self.RSTDP
        else:
            self.procedure = self.STDP

        synapse.src_spike_history = pymo.np.array(
            [pymo.np.zeros(synapse.src.size) for _ in range(synapse.time_window)]
        )
        synapse.dst_spike_history = pymo.np.array(
            [pymo.np.zeros(synapse.dst.size) for _ in range(synapse.time_window)]
        )

        synapse.cos_sim_over_time = [
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 1])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 1])
                )
            )
        ]
        synapse.cos_sim_over_time2 = [
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 2])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 2])
                )
            )
        ]
        synapse.cos_sim_over_time3 = [
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 2])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 2])
                )
            )
        ]
        synapse.cos_sim_over_time4 = [
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        ]
        synapse.cos_sim_over_time5 = [
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        ]
        synapse.cos_sim_over_time6 = [
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        ]
        synapse.cos_sim_over_time7 = [
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        ]
        synapse.cos_sim_over_time8 = [
            (
                pymo.np.dot(synapse.W[:, 2], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 2])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        ]
        synapse.cos_sim_over_time9 = [
            (
                pymo.np.dot(synapse.W[:, 2], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 2])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        ]
        synapse.cos_sim_over_time0 = [
            (
                pymo.np.dot(synapse.W[:, 3], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 3])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        ]
        return

    def forward(self, synapse):
        synapse.src_spike_history = pymo.np.insert(
            synapse.src_spike_history, 0, synapse.src.spike, axis=0
        )[:-1]
        synapse.dst_spike_history = pymo.np.insert(
            synapse.dst_spike_history, 0, synapse.dst.spike, axis=0
        )[:-1]
        synapse.cos_sim_over_time.append(
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 1])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 1])
                )
            )
        )
        synapse.cos_sim_over_time2.append(
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 2])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 2])
                )
            )
        )
        synapse.cos_sim_over_time3.append(
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 2])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 2])
                )
            )
        )
        synapse.cos_sim_over_time4.append(
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        )
        synapse.cos_sim_over_time5.append(
            (
                pymo.np.dot(synapse.W[:, 0], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 0])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        )
        synapse.cos_sim_over_time6.append(
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        )
        synapse.cos_sim_over_time7.append(
            (
                pymo.np.dot(synapse.W[:, 1], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 1])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        )
        synapse.cos_sim_over_time8.append(
            (
                pymo.np.dot(synapse.W[:, 2], synapse.W[:, 3])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 2])
                    * pymo.np.linalg.norm(synapse.W[:, 3])
                )
            )
        )
        synapse.cos_sim_over_time9.append(
            (
                pymo.np.dot(synapse.W[:, 2], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 2])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        )
        synapse.cos_sim_over_time0.append(
            (
                pymo.np.dot(synapse.W[:, 3], synapse.W[:, 4])
                * 100.00
                / (
                    pymo.np.linalg.norm(synapse.W[:, 3])
                    * pymo.np.linalg.norm(synapse.W[:, 4])
                )
            )
        )
        self.procedure(synapse)
        self.curr_time += 1
        # print(self.curr_time, synapse.W)
        return

    def STDP(self, synapse):
        for i in range(synapse.src.size):
            if synapse.src.spike[i] == True:
                for time_window in range(synapse.dst_spike_history.shape[0]):
                    for j in range(synapse.dst_spike_history.shape[1]):
                        if synapse.dst_spike_history[time_window, j] == True:
                            new_weight = synapse.W[i, j] + self.learningFunction(
                                -time_window
                            )
                            if new_weight > self.max_weight:
                                synapse.W[i, j] = self.max_weight
                            elif new_weight < self.min_weight:
                                synapse.W[i, j] = self.min_weight
                            else:
                                synapse.W[i, j] = new_weight

        for i in range(synapse.dst.size):
            if synapse.dst.spike[i] == True:
                for time_window in range(synapse.src_spike_history.shape[0]):
                    for j in range(synapse.src_spike_history.shape[1]):
                        if synapse.src_spike_history[time_window, j] == True:
                            new_weight = synapse.W[j, i] + self.learningFunction(
                                time_window
                            )
                            if new_weight > self.max_weight:
                                synapse.W[j, i] = self.max_weight
                            elif new_weight < self.min_weight:
                                synapse.W[j, i] = self.min_weight
                            else:
                                synapse.W[j, i] = new_weight

    def RSTDP(self, synapse):
        for i in range(synapse.src.size):
            if synapse.src.spike[i] == True:
                for time_window in range(synapse.dst_spike_history.shape[0]):
                    for j in range(synapse.dst_spike_history.shape[1]):
                        if synapse.dst_spike_history[time_window, j] == True:
                            # print(time_window)
                            new_weight = synapse.W[i, j] + self.learningFunction(
                                -time_window - 1
                            )
                            if new_weight > self.max_weight:
                                synapse.W[i, j] = self.max_weight
                            elif new_weight < self.min_weight:
                                synapse.W[i, j] = self.min_weight
                            else:
                                synapse.W[i, j] = new_weight

        for i in range(synapse.dst.size):
            if synapse.dst.spike[i] == True:
                for time_window in range(synapse.src_spike_history.shape[0]):
                    for j in range(synapse.src_spike_history.shape[1]):
                        if synapse.src_spike_history[time_window, j] == True:
                            new_weight = synapse.W[j, i]
                            if synapse.reward[i, self.curr_time] == 2:
                                new_weight = synapse.W[j, i] + self.learningFunction(
                                    -time_window - 1
                                )
                            elif synapse.reward[i, self.curr_time] == 1:
                                new_weight = synapse.W[j, i] + self.learningFunction(
                                    time_window + 1
                                )

                            if new_weight > self.max_weight:
                                synapse.W[j, i] = self.max_weight
                            elif new_weight < self.min_weight:
                                synapse.W[j, i] = self.min_weight
                            else:
                                synapse.W[j, i] = new_weight

    @staticmethod
    def STDP_delta_function(t):
        if t > 0:
            return 2 * pymo.np.exp(-t / 5)
        elif t < 0:
            return -0.66 * pymo.np.exp(t / 20)  # Equilibrium at -0.703490
        else:
            return 0

    @staticmethod
    def flatSTDP_delta_function(t):
        if t > 0:
            return 0.6
        elif t < 0:
            return -0.6
        else:
            return 0
