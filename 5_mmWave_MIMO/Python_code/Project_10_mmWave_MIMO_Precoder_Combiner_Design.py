import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
import MIMO

# simulation parameters
t = 32; r = 32; 
numRF = 6; 
G = 64;    
L = 8;  
Ns = 6; 
ITER = 100; 

# Initializations
SNRdB = np.arange(-5,6,1); 
C_HYB = np.zeros(len(SNRdB));
C_MIMO = np.zeros(len(SNRdB)); 

# G-quantized Txarray response matrix
A_T = MIMO.ArrayDictionary(G,t);
A_R = MIMO.ArrayDictionary(G,r);

for ix in range(ITER): 
    print(ix);
    
    # Channel generation
    tax = np.random.choice(G, L, replace=False);
    rax = np.random.choice(G, L, replace=False);
    chGain = 1/np.sqrt(2)*(nr.normal(0,1,L)+1j*nr.normal(0,1,L));
    A_T_channel = A_T[:, tax];
    A_R_channel = A_R[:, rax];
    H = np.sqrt(t*r/L)*nl.multi_dot([A_R_channel,np.diag(chGain),MIMO.H(A_T_channel)]);

    U, S, VH = nl.svd(H, full_matrices=True)

    V = MIMO.H(VH)
    Fopt = V[:,0:Ns]; 
    FBB, FRF = MIMO.SOMP(Fopt, A_T, np.identity(t), numRF);
    FBB_NORM = FBB*np.sqrt(Ns)/nl.norm(np.matmul(FRF,FBB)); 
    for cx in range(len(SNRdB)):
        npow = 10**(-SNRdB[cx]/10);
        mmseINV = nl.inv(MIMO.AHA(np.matmul(H,Fopt)) + npow*Ns*np.identity(Ns));
        Wmmse_opt = nl.multi_dot([H,Fopt,mmseINV]);
        C_MIMO[cx] = C_MIMO[cx] + \
        MIMO.mimo_capacity(nl.multi_dot([MIMO.H(Wmmse_opt),H,Fopt]), 1/Ns*np.identity(Ns), npow*MIMO.AHA(Wmmse_opt));
        HFp = nl.multi_dot([H,FRF,FBB_NORM]);
        Ryy = 1/Ns*MIMO.AAH(HFp) + npow*np.identity(r);
        Wmmse_Hyb = np.matmul(HFp,nl.inv(MIMO.AHA(HFp) + npow*Ns*np.identity(Ns)));
        WBB, WRF = MIMO.SOMP(Wmmse_Hyb, A_R, Ryy, numRF);
        C_HYB[cx] = C_HYB[cx] + \
            MIMO.mimo_capacity(nl.multi_dot([MIMO.H(WBB), MIMO.H(WRF),H,FRF,FBB_NORM]), 1/Ns*np.identity(Ns), npow*MIMO.AHA(np.matmul(WRF,WBB)));
        
C_MIMO = C_MIMO/ITER; C_HYB = C_HYB/ITER;
plt.plot(SNRdB, C_MIMO,'r-s');
plt.plot(SNRdB, C_HYB,'b^-.');
plt.grid(1,which='both')
plt.legend(["Ideal Digital", "Hybrid Precoder"], loc ="lower right");
plt.suptitle('Capacity vs SNR for mmWave MIMO')
plt.ylabel('Capacity (b/s/Hz)')
plt.xlabel('SNRdB') 
