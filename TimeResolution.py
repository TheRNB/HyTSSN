import pymonntorch as pymo

class TimeResolution(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, network):
        network.dt = self.parameter("dt", default=1) # MILISECOND
        network.time_passed = 0
        return

    def forward(self, network):
        network.time_passed += network.dt
        return

