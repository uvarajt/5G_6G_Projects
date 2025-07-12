import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import MIMO


SNRdB = np.arange(-10,10,1); # Signal-to-noise power ratio in dB
SNR = 10**(SNRdB/10); # Signal-to-noise power ratio
numBlocks = 100; # Number of blocks
Capacity_OPT = np.zeros(len(SNRdB)); # Optimal capacity of MIMO channel 
Capacity_EQ = np.zeros(len(SNRdB));# Capacity with equal power allocation
r = 4; # Number of receive antennas
t = 4; # Number of transmit antennas



for L in range(numBlocks): # Looping over blocks
    # r x t MIMO channel matrix H
    H = (nr.normal(0.0, 1.0,(r,t))+1j*nr.normal(0.0, 1.0,(r,t)))/np.sqrt(2)
    for kx in range(len(SNRdB)): #Loop over SNR values
        #Call routine for optimal MIMO capacity calculation
        Capacity_OPT[kx] = Capacity_OPT[kx] + MIMO.OPT_CAP_MIMO(H,SNR[kx]);
        #Call routine for MIMO capacity calculation with equal power allocation
        Capacity_EQ[kx] = Capacity_EQ[kx] + MIMO.EQ_CAP_MIMO(H,SNR[kx]);

                                                           
#Averaging capacity values over the blocks
Capacity_OPT = Capacity_OPT/numBlocks;
Capacity_EQ = Capacity_EQ/numBlocks;

# Plotting capacities for equal and optimal power allocation obtained via simulation
plt.plot(SNRdB,Capacity_OPT,'b-s');
plt.plot(SNRdB,Capacity_EQ,'r-.o');
plt.grid(1,which='both')
plt.legend(["OPT","Equal"], loc ="upper left");
plt.suptitle('MIMO Capacity vs SNR(dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('Capacity (b/s/Hz)') 

