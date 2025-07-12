import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import numpy.linalg as nl
import MIMO

Nr = 4; Nt = 4;
num_elements = np.arange(2,41,2); 
K = 25; 
nBlocks = 40;
Pt_dBm = 30; 
Pt = 10**((Pt_dBm-30)/10);
No_dBm = -90; 
No = 10**((No_dBm-30)/10);
H = 10; 
dDbar = 600; 
dpbar = 2;
dhbar = 2;

dD = np.sqrt(dDbar**2 + H**2); 
dBI = np.sqrt((dDbar - dhbar)**2 + dpbar**2); 
dIU = np.sqrt(dhbar**2 + dpbar**2 + H**2); 
beta0_dB = -30;
beta0 =  10**(beta0_dB/10);
alphaD = 3.5; 
alphaBI = 1.5; 
alphaIU = 1.5; 
betaD = beta0/(dD**alphaD); 
betaBI = beta0/(dBI**alphaBI); 
betaIU = beta0/(dIU**alphaIU); 
CAP_AO = np.zeros(len(num_elements)); 
CAP_random = np.zeros(len(num_elements)); 
CAP_withoutIRS = 0;

for blk in range(nBlocks):
    print(blk)
    for m in range(len(num_elements)):       
        M = num_elements[m]
        H = np.sqrt(0.5*betaD)*(nr.normal(0.0, 1.0,(Nr,Nt)) + 1j*nr.normal(0.0, 1.0,(Nr,Nt)));
        T = np.sqrt(0.5*betaBI)*(nr.normal(0.0, 1.0,(M,Nt)) + 1j*nr.normal(0.0, 1.0,(M,Nt))); 
        R = np.sqrt(0.5*betaIU)*(nr.normal(0.0, 1.0,(Nr,M)) + 1j*nr.normal(0.0, 1.0,(Nr,M)));      
        
        alpha = np.ones(M)+1j*np.zeros(M);
        Q = (Pt/Nt)*np.identity(Nt);
        Q_sqrt=np.sqrt(Pt/Nt)*np.identity(Nt);
        H_dash = nl.multi_dot([H,Q_sqrt]);
        T_dash = nl.multi_dot([T,Q_sqrt]);
        
        for kk in range(K):
            alpha = MIMO.OPT_REFL_COEFF(H_dash, R, T_dash, alpha, No, Nr, Nt, M);
            H_tilde = H + nl.multi_dot([R,np.diag(alpha),T]);
            Q,Q_sqrt,CAP = MIMO.OPT_Q_MIMO(H_tilde, Pt, No);
            
            #UQ,SigmaQ,VQ = nl.svd(Q_sqrt,full_matrices=False);

            H_dash = nl.multi_dot([H,Q_sqrt]);
            T_dash = nl.multi_dot([T,Q_sqrt]);
        CAP_AO[m] = CAP_AO[m]+ CAP;        
        
        alpha = np.exp(1j*2*np.pi*nr.random(M));
        H_tilde = H + nl.multi_dot([R,np.diag(alpha),T]);
        Q, Q_sqrt, CAP = MIMO.OPT_Q_MIMO(H_tilde,Pt,No);
        CAP_random[m] = CAP_random[m] + CAP;
        
    Q, Q_sqrt, CAP = MIMO.OPT_Q_MIMO(H,Pt,No);
    CAP_withoutIRS = CAP_withoutIRS + CAP;


CAP_AO = CAP_AO/nBlocks;
CAP_random = CAP_random/nBlocks;
CAP_withoutIRS = CAP_withoutIRS/nBlocks;
plt.yscale('log')
plt.plot(num_elements,np.real(CAP_AO),'g-p',linewidth=2, markersize=9);
plt.plot(num_elements,np.real(CAP_random),'b-.s',linewidth=2, markersize=9);
plt.plot(num_elements,np.real(CAP_withoutIRS)*np.ones(len(num_elements)),'r:o',linewidth=2, markersize=9);
plt.grid(1,which='both')
plt.suptitle('Rate vs number of reflecting elements for MIMO-IRS system')
plt.legend(["AO","Random phase", "Without IRS"], loc ="upper left");
plt.xlabel('Number of reflecting elements N')
plt.ylabel('Rate (bps/Hz)') 
