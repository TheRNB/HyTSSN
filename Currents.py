import pymonntorch as pymo
import math

class ConstantCurrent(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.value = self.parameter("current", default=None, required=True)
        self.noise = self.parameter("noise", default=False, required=False)
        neural_group.I = neural_group.vector(self.value)
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return
        
    def forward(self, neural_group):
        neural_group.I = neural_group.vector(self.value)
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return


class StepCurrent(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.value = self.parameter("initial_current", default=None, required=True)
        self.step_threshold = self.parameter("step_threshold", default=None, required=True)
        self.step = self.parameter("final_current", default=None, required=True)
        self.noise = self.parameter("noise", default=False, required=False)
        
        neural_group.I = neural_group.vector(self.value)
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return

    def forward(self,  neural_group):
        if neural_group.network.time_passed <= self.step_threshold:
            neural_group.I = neural_group.vector(self.value)
        else:
            neural_group.I = neural_group.vector(self.step)
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return


class MultiStepCurrent(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.value = self.parameter("initial_current", default=None, required=True)
        self.step = self.parameter("step", default=None, required=True)
        self.final = self.parameter("final_current", default=None, required=True)
        self.noise = self.parameter("noise", default=False, required=False)
        
        neural_group.I = neural_group.vector(self.value)
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return

    def forward(self,  neural_group):
        #neural_group.I = min(neural_group.I + neural_group.vector(self.step), neural_group.vector(self.final))
        neural_group.I = neural_group.I + neural_group.vector(self.step)
        neural_group.I[neural_group.I > self.final] = self.final
        if self.noise:
            neural_group.I += pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return


class SineCurrent(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.value = self.parameter("initial_current", default=None, required=True)
        self.modifier = self.parameter("frequency_modifier", default=None, required=True)
        self.coefficient = self.parameter("current_modifier", default=None, required=True)
        self.noise = self.parameter("noise", default=False, required=False)

        neural_group.I = neural_group.vector(self.value)
        if self.noise:
            neural_group.I = neural_group.I + pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return

    def forward(self, neural_group):
        neural_group.I = neural_group.vector(self.value + (self.coefficient * math.sin(self.modifier * neural_group.network.time_passed)))
        if self.noise:
            neural_group.I = neural_group.I + pymo.torch.normal(mean=0.0, std=1, size=neural_group.I.size())
        return

