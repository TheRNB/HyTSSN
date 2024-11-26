import pymonntorch as pymo
import numpy as np
import Models
import TimeResolution
import Currents
import Synapse
import Dendrite
import Input
import Encoder

pymo.torch.manual_seed(43)

input1 = Encoder.poissonValues(
    image=np.array([0, 0, 0, 201, 123, 241, 125, 88]),
    steps=15,
    timeframe=20
)

input2 = Encoder.poissonValues(
    image=np.array([178, 160, 0, 0, 0, 241, 125, 88]),
    steps=15,
    timeframe=20
)

input3 = Encoder.poissonValues(
    image=np.array([178, 160, 193, 201, 0, 0, 0, 88]),
    steps=15,
    timeframe=20
)

input4 = Encoder.poissonValues(
    image=np.array([0, 160, 193, 201, 123, 241, 0, 0]),
    steps=15,
    timeframe=20
)

input5 = Encoder.poissonValues(
    image=np.array([178, 160, 193, 201, 123, 241, 125, 88]),
    steps=15,
    timeframe=20
)

zero = np.zeros((input1.shape[0], 20))
input = np.copy(input1)
input = np.concatenate((input, input1), axis=1)
input = np.concatenate((input, input1), axis=1)
input = np.concatenate((input, zero), axis=1)
input = np.concatenate((input, input2), axis=1)
input = np.concatenate((input, input2), axis=1)
input = np.concatenate((input, input2), axis=1)
input = np.concatenate((input, zero), axis=1)
input = np.concatenate((input, input3), axis=1)
input = np.concatenate((input, input3), axis=1)
input = np.concatenate((input, input3), axis=1)
input = np.concatenate((input, zero), axis=1)
input = np.concatenate((input, input4), axis=1)
input = np.concatenate((input, input4), axis=1)
input = np.concatenate((input, input4), axis=1)
input = np.concatenate((input, zero), axis=1)
input = np.concatenate((input, input5), axis=1)
input = np.concatenate((input, input5), axis=1)
input = np.concatenate((input, input5), axis=1)

reward = np.array([
    np.concatenate((np.ones(60)*1, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(120))),
    np.concatenate((np.ones(60)*2, np.zeros(20), np.ones(60)*1, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(120))),
    np.concatenate((np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*1, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(120))),
    np.concatenate((np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*1, np.zeros(20), np.ones(60)*2, np.zeros(120))),
    np.concatenate((np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*2, np.zeros(20), np.ones(60)*1, np.zeros(120))),
])

#print(reward)
for i in range(2):
    network = pymo.Network(device="cpu", synapse_mode="SxD", dtype=pymo.torch.float64, behavior={
        2: TimeResolution.TimeResolution(dt=1)
    }, tag="main_net")

    ng1 = pymo.NeuronGroup(net=network, size=5, behavior={
        10: Currents.ConstantCurrent(
            current = 5,
            noise = False
        ),
        17: Dendrite.Dendrite(
        ),
        20: Models.LIF(
            R = 5,
            threshold = -37,
            u_rest = -67,
            u_reset = -75,
            tau = 10,
            refractory = False,
            #kwta = 1
        ),
        #21: Models.acticityHomeostasis(
        #    window_size = 10,
        #    activity_rate = 1,
        #    updating_rate = 0.95,
        #    decay_rate = 1.00
        #),
        #21: Models.voltageHomeostasis(
        #    activity_rate = -37,
        #    max_tao = 0,
        #    min_tao = -80,
        #    eta = 0.001
        #),
        100: pymo.Recorder(
            variables = ["torch.sum(I)", "u", "I"],
            tag = "ng1_recorder"
        ),
        101: pymo.EventRecorder(
            variables = ["spike"],
            tag = "ng1_eventrecorder"
        )
    }, tag="ng1")

    ng2 = pymo.NeuronGroup(net=network, size=8, behavior={
        10: Input.Input(
            matrix = input
        )
    }, tag="ng2")

    syn1 = pymo.SynapseGroup(net=network, src=ng2, dst=ng1, behavior={
        25: Synapse.all_to_all_connection(
            J0 = 10,
            std = 0.10,
            pre_is_excitate = True,
        ),
        30: Synapse.Learning(
            function = "normal",
            reward = reward,
            time = 20,
            procedure = "rstdp"
        ),
        100: pymo.Recorder(
            variables = ["W"],
            tag = "sg1_recorder"
        ),
    }, tag="syn_ng1_ng2")

    syn2 = pymo.SynapseGroup(net=network, src=ng1, dst=ng1, behavior={
        25: Synapse.all_to_all_connection(
            J0 = 1,
            std = 0.10,
            pre_is_excitate = False,
        ),
        100: pymo.Recorder(
            variables = ["W"],
            tag = "sg2_recorder"
        ),
    }, tag="syn_ng1_ng1")
    
    network.initialize(False)

    #network.simulate_iteration()
    network.simulate_iterations(400)

import matplotlib.pyplot as plt

def clear_plot(save_location, dpi=300):
    plt.savefig(save_location, dpi=dpi)
    plt.clf()
    return
def que_plot(variables, xlabel, ylabel, title, color=None):
    if color == None:
        plt.plot(variables)
    else:
        plt.plot(variables, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return

def que_scatter(variableX, variableY, xlabel, title, colors=None, size=None, ylabel=None):
    plt.scatter(variableX, variableY, c = colors, s=size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return

que_plot(syn1.cos_sim_over_time, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time2, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time4, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time5, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time3, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time6, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time7, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time8, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time9, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 2 output neurons")
que_plot(syn1.cos_sim_over_time0, "time", "cos sim (%)", "Cosine Similarity of weight vectors for 5 output neurons")
plt.legend(["#0 w/ #1", "#0 w/ #2", "#1 w/ #2", "#0 w/ #3", "#0 w/ #4", "#1 w/ #2", "#1 w/ #3", "#1 w/ #4", "#2 w/ #3", "#2 w/ #4", "#3 w/ #4"])
#plt.legend(["Flat R-STDP", "R-STDP"])
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_cos.jpg", dpi=600)

colors = ["GREEN","RED","PINK","BLUE","ORANGE","GRAY","PURPLE", "YELLOW"]
plt.figure(figsize=(10,9))
for i in range(ng2.size):
    que_plot(variables=network["sg1_recorder", 0].variables["W"][:,i,:],
            xlabel="time", ylabel="synaptic weight", title="synaptic weights over time for neurons ", color=colors[i])
plt.legend(["Neuron #"+str(i)+" to #"+str(j) for i in range(ng2.size) for j in range(ng1.size)])
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_syn.jpg", dpi=600)

que_plot(variables=network["ng1_recorder", 0].variables["I"][:,:], 
         xlabel="time", ylabel="I(current)", title="I-t for Destination")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_It.jpg", dpi=600)

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
colors = np.random.uniform(15, 80, (network["ng1_eventrecorder", 0].variables["spike"]).size()[0])
"""que_scatter(variableX=x, variableY=np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]),
            colors=colors,
            xlabel="time", title='spike time pattern for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_stp.jpg")"""

x_new = [x[i+1]-x[i] for i in range(len(x)-1)]
"""que_scatter(variableX=x_new, variableY=max(0,np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1)),
            colors=colors[:-1],
            xlabel='time', title='Spike time interval for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_sti.jpg")"""

dot_size = []
x_new_sorted = sorted(x_new)
for i in range(len(x_new_sorted)):
    if i == 0 or x_new_sorted[i-1]!=x_new_sorted[i]:
        dot_size.append(10)
    else:
        dot_size.append(dot_size[-1]+10)
"""que_scatter(variableX=x_new_sorted, variableY=np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1),
            colors=colors[:-1], size=dot_size,
            xlabel='time', title='Cumulative spike time interval for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_csti.jpg")"""

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
y = [0 for _ in range(network["ng1_recorder", 0].variables["u"][:,0].shape[0]+1)]
for num in x:
    y[num] += 1
y = np.array(y)
que_plot(variables=y/ng1.size,
         xlabel="time", ylabel="Activity", title="Activity for population #1")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_A.jpg")

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,:].cpu().numpy()
x_raster, y_raster = [], []
for time, id in x:
    x_raster.append(time)
    y_raster.append(id)
x_raster, y_raster = np.array(x_raster), np.array(y_raster)
plt.figure(figsize=(10,2))
que_scatter(variableX=x_raster, variableY=y_raster,
            colors="red", size=1,
            xlabel='time', ylabel='Neurons', title="Raster Plot")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_raster.jpg", dpi=600)

x_raster, y_raster = [], []
for i in range(input.shape[0]):
    for j in range(input.shape[1]):
        if input[i,j] == 1:
            x_raster.append(j)
            y_raster.append(i)
x_raster, y_raster = np.array(x_raster), np.array(y_raster)
plt.figure(figsize=(10,2))
plt.scatter(x_raster, y_raster, c = "DARKGREEN", s=1)
plt.xlabel("time")
plt.ylabel("neurons")
plt.title(" Input Pattern ")
plt.ylim(-1, input.shape[0])
plt.xlim(-1, 400)
plt.savefig("/Users/aaron/Downloads/Figure_1_input.jpg", dpi=600)
plt.clf()

print((network["ng1_eventrecorder", 0].variables["spike"]).size())
