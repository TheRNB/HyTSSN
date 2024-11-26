import pymonntorch as pymo

class Input(pymo.Behavior):
    #TODO: migrate to PyTorch
    def initialize(self, neural_group):
        self.input = self.parameter("matrix", default=None, required=True)
        self.iterationCount = 1

        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                if self.input[i, j] == 0:
                    self.input[i, j] = False
                else:
                    self.input[i, j] = True

        self.timewidth = self.input.shape[1]
        neural_group.spike = self.input[:,1]
        return

    def forward(self, neural_group):
        if self.timewidth > self.iterationCount:
            neural_group.spike = self.input[:,self.iterationCount]
            #print("in time ", self.iterationCount, " set neurons to ", neural_group.spike)
            self.iterationCount += 1
        return
    