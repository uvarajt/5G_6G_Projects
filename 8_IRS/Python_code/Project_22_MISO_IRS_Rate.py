
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import numpy.linalg as nl

Nr = 1; 
Nt = 4;
num_elements = np.arange(2,41,2); 
K = 20; 
nBlocks = 100;
Pt_dBm = 30;
Pt = 10**((Pt_dBm-30)/10);
No_dBm = -90; 
No = 10**((No_dBm-30)/10);
H_sd = 10; 
dDbar = 600; 
dpbar = 2; 
dhbar = 2; 

dD = np.sqrt(dDbar**2 + H_sd**2); 
dBI = np.sqrt((dDbar - dhbar)**2 + dpbar**2); 
dIU = np.sqrt(dhbar**2 + dpbar**2 + H_sd**2); 
beta0_dB = -30;
beta0 =  10**(beta0_dB/10);
alphaD = 3.5; 
alphaBI = 1.5; 
alphaIU = 1.5; 
betaD = beta0/(dD**alphaD); 
betaBI = beta0/(dBI**alphaBI); 
betaIU = beta0/(dIU**alphaIU); 
rate_AO = np.zeros(len(num_elements));
rate_random = np.zeros(len(num_elements));
rate_withoutIRS = 0;

for blk in range(nBlocks):
    h_sd = np.sqrt(0.5*betaD)*(nr.normal(0.0, 1.0,(Nt,1))+1j*nr.normal(0.0, 1.0,(Nt,1))); 
    w = np.sqrt(Pt)*h_sd/nl.norm(h_sd);
    gamma =  np.squeeze(np.abs(np.matmul(np.conj(h_sd.T),w))**2/No);
    rate_withoutIRS = rate_withoutIRS + np.log2(1+gamma);

    for m in range(len(num_elements)):
        N = num_elements[m]
        H_sr = np.sqrt(0.5*betaBI)*(nr.normal(0.0, 1.0,(N,Nt)) + 1j*nr.normal(0.0, 1.0,(N,Nt)));
        h_rd = np.sqrt(0.5*betaIU)*(nr.normal(0.0, 1.0,(N,1)) + 1j*nr.normal(0.0, 1.0,(N,1)));
        
        theta = 2*np.pi*nr.random(N);
        Theta = np.diag(np.exp(1j*theta));
        h_eff = nl.multi_dot([np.conj(h_rd.T),Theta,H_sr]) + np.conj(h_sd.T);
        w = np.sqrt(Pt)*np.conj(h_eff.T)/nl.norm(h_eff);
        gamma = np.abs(np.matmul(h_eff,w))**2/No;
        rate_random[m] = rate_random[m] + np.log2(1+np.squeeze(gamma));

        Phi_0 = np.angle(np.matmul(np.conj(h_sd.T),w));
        
        for kk in range(K):
            theta = np.zeros(N);
            for n in range(N):
                theta[n] = Phi_0 + np.angle(h_rd[n]) - np.angle(np.matmul(H_sr[n,:],w));
                
            Theta = np.diag(np.exp(1j*theta));
            h_eff = nl.multi_dot([np.conj(h_rd.T),Theta,H_sr]) + np.conj(h_sd.T);

            w=np.sqrt(Pt)*np.conj(h_eff.T)/nl.norm(h_eff);
            Phi_0 = np.angle(np.matmul(np.conj(h_sd.T),w));
 
        gamma = np.abs(np.matmul(h_eff,w))**2/No;
        rate_AO[m] = rate_AO[m] + np.log2(1+np.squeeze(gamma));
        
rate_AO = rate_AO/nBlocks;
rate_random = rate_random/nBlocks;
rate_withoutIRS = rate_withoutIRS/nBlocks;

plt.yscale('log')
plt.plot(num_elements,rate_AO,'g-p',linewidth=2, markersize=9);
plt.plot(num_elements,rate_random,'b-.s',linewidth=2, markersize=9);
plt.plot(num_elements,rate_withoutIRS*np.ones(len(num_elements)),'r:o',linewidth=2, markersize=9);
plt.grid(1,which='both')
plt.suptitle('Rate vs number of reflecting elements for MISO-IRS system')
plt.legend(["AO","Random phase", "Without IRS"], loc ="upper left");
plt.xlabel('Number of reflecting elements N')
plt.ylabel('Rate (bps/Hz)') 

