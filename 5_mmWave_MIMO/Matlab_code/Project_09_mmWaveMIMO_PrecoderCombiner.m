close all; 
clear all; 
rng('shuffle'); 

t = 32; r = 32; % No. of Tx and Rx antennas
numRF = 6; % No. of RF chains
G = 64; % Size of dictionaries
L = 8; % No. of multipath components
Ns = 6; % No. of symbols
ITER = 1000; % No. of iterations

SNRdB = -5:1:5; % SNR in dB
C_HYB = zeros(length(SNRdB),1); 
C_MIMO = zeros(length(SNRdB),1);

A_T = 1/sqrt(t)*exp(-j*pi*[0:t-1]'*[2/G*[0:G-1] - 1]); 
A_R = A_T;


for iter1 = 1:ITER 
    % Channel Generator
    AoDlist = randperm(G,L); AoAlist = randperm(G,L);  
    chGain = 1/sqrt(2)*(randn(L,1) + j*randn(L,1));
    H = sqrt(t*r/L)*A_R(:,AoAlist)*diag(chGain)*A_T(:,AoDlist)';    

    [U,S,V] = svd(H);
    Fopt = V(:,1:Ns); 
    [FBB, FRF] = SOMP_mmW_precoder(Fopt, A_T, eye(t), numRF);
    FBB_NORM = sqrt(Ns)/(norm(FRF*FBB,'fro'))*FBB; 

    for i_snr = 1:length(SNRdB)
        np = 10^(-SNRdB(i_snr)/10); 

        Wmmse_opt = H*Fopt*inv(Fopt'*H'*H*Fopt + np*Ns*eye(Ns));
        C_MIMO(i_snr) = C_MIMO(i_snr) + mimo_capacity(Wmmse_opt'*H*Fopt, 1/Ns*eye(Ns), np*Wmmse_opt'*Wmmse_opt);

        Ryy = 1/Ns*H*FRF*FBB_NORM*FBB_NORM'*FRF'*H' + np*eye(r);
        Wmmse_Hyb = H*FRF*FBB_NORM*inv(FBB_NORM'*FRF'*H'*H*FRF*FBB_NORM + np*Ns*eye(Ns));
        [WBB, WRF] = SOMP_mmW_precoder(Wmmse_Hyb, A_R, Ryy, numRF);
        C_HYB(i_snr) = C_HYB(i_snr) + mimo_capacity(WBB'*WRF'*H*FRF*FBB_NORM, 1/Ns*eye(Ns), np*WBB'*WRF'*WRF*WBB);
     end
end

C_MIMO = C_MIMO/ITER; C_HYB = C_HYB/ITER;

plot(SNRdB,abs(C_MIMO),'b - s','linewidth',3.0,'MarkerFaceColor','b','MarkerSize',9.0);
hold on;
plot(SNRdB,abs(C_HYB),'m -. o','linewidth',3.0,'MarkerFaceColor','m','MarkerSize',9.0);
grid on; 
axis tight;
xlabel('SNR(dB)'); 
ylabel('Capacity (b/s/Hz)');
legend('Conventional MIMO','Hybrid MIMO','Location','NorthWest'); 
title('mmWave MIMO Capacity vs SNR');
