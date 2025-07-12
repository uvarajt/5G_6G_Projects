
clear all
close all

beta1=5; beta2=1; 
SNRdB=0:2:30; SNR=10.^(SNRdB/10);    

a1=0.9; a2=0.1; % Power allocation factor
 
blklen=1000000;
tildeR_1=1; tildeR_2=1; % Desired data rates                
R1=2^(tildeR_1)-1;
R2=2^(tildeR_2)-1;
phi2=(R1/(a1*beta1))+(R2/(a2*beta2))+((R1*R2)/(a1*beta1));
phi3=1+((R1*a2*beta2)/(a1*beta1));


for ix=1:length(SNR)
    ix
    outage1=0; outage2=0;
    rhos=SNR(ix);
    
    h1=sqrt(beta1/2)*(randn(1,blklen)+1j*randn(1,blklen));    
    h2=sqrt(beta2/2)*(randn(1,blklen)+1j*randn(1,blklen));
    b_1 = abs(h1).^2;
    b_2 = abs(h2).^2;   
    
    gamma1 = (a1*rhos*b_1)./(a2*rhos*b_2+1)
    gamma2 = a2*rhos*b_2;

    outage1 = (log2(1+gamma1)<tildeR_1); %Simulated outage values
    outage2 = ((log2(1+gamma1)<tildeR_1) | (log2(1+gamma2)<tildeR_2));

    Pout1(ix) = sum(outage1)/blklen;
    Pout2(ix) = sum(outage2)/blklen;
    
end 
% Theoretical outage values
Pout1_theory = 1- exp(-R1./(a1*SNR*beta1))/(1+R1*a2*beta2/(a1*beta1));
Pout2_theory = 1- exp(-phi2./SNR)/phi3;


semilogy(SNRdB,Pout1,'r -','Linewidth',2.0)
hold on
semilogy(SNRdB,Pout2,'m -','Linewidth',2.0,'markerfacecolor','m')
semilogy(SNRdB,Pout1_theory,'r s','Linewidth',2.0,'markerfacecolor','r')
semilogy(SNRdB,Pout2_theory,'m o','Linewidth',2.0,'markerfacecolor','m')
grid on
legend('Outage User 1 (Sim.)','Outage User 2 (Sim.)','Outage User 1 (Theory)','Outage User 2 (Theory)')
xlabel('SNR (dB)')
ylabel('Prob of Outage')
title('Pout vs SNR for UL NOMA')