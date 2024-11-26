import pymonntorch as pymo

class Dendrite(pymo.Behavior):
    #TODO: migrate to PyTorch
    def forward(self, neural_group):
        for synapse in neural_group.afferent_synapses["All"]: #Efferent keeps the output synapses
            #print(synapse.I)
            neural_group.I += synapse.I

