import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt

num_elements = np.arange(1,41,2);     
nBlocks = 100;
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

rate_Random = np.zeros(len(num_elements));
rate_Opt = np.zeros(len(num_elements));
rate_withoutIRS = np.zeros(len(num_elements));
      
for blk in range(nBlocks):
    h_sd = np.sqrt(0.5*betaD)*(nr.normal(0,1)+1j*nr.normal(0,1));       
    for ix in range(len(num_elements)):
        N = num_elements[ix];
        SNR=Pt/No;
        h_rd = np.sqrt(0.5*betaIU)*(nr.normal(0.0, 1.0,(N,1))+1j*nr.normal(0.0, 1.0,(N,1)));  
        h_sr = np.sqrt(0.5*betaBI)*(nr.normal(0.0, 1.0,(N,1))+1j*nr.normal(0.0, 1.0,(N,1)));  
        r1 = np.conj(h_sd);
        r2 = np.conj(h_sd);  
        for n in range(N):
            Theta_rand = 2*np.pi*nr.random();
            Theta_opt = (-np.angle(h_sd)+np.angle(h_rd[n])-np.angle(h_sr[n]))%(2*np.pi);
 
            r1 = r1 + np.conj(h_rd[n])*h_sr[n]*np.exp(1j*Theta_rand)+ np.conj(h_sd);
            r2 = r2 + np.conj(h_rd[n])*h_sr[n]*np.exp(1j*Theta_opt)+ np.conj(h_sd);

        rate_Random[ix] = rate_Random[ix] + np.log2(1+SNR*np.abs(r1[0])**2);
        rate_Opt[ix] = rate_Opt[ix] + np.log2(1+SNR*np.abs(r2[0])**2);
    rate_withoutIRS = rate_withoutIRS+ np.log2(1+SNR*np.abs(h_sd)**2);

rate_Random=rate_Random/nBlocks;
rate_Opt=rate_Opt/nBlocks;
rate_withoutIRS=rate_withoutIRS/nBlocks;

plt.yscale('log')
plt.plot(num_elements,rate_Random,'g-p');
plt.plot(num_elements,rate_Opt,'b-.s');
plt.plot(num_elements,rate_withoutIRS,'r:o');
plt.grid(1,which='both')
plt.suptitle('SISO IRS Rate vs number of elements')
plt.legend(["Random phase shift","Optimal phase shift", "without IRS"], loc ="center right");
plt.xlabel('Number of elements')
plt.ylabel('Rate (bps/Hz)') 
