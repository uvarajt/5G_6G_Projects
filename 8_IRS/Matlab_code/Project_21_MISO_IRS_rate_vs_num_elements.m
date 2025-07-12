clc; clear all; close all;

Nr = 1; % Number of antennas at the UE
Nt = 16; % Number of antennas at the BS
num_elements = 2:2:40; % Number of IRS elements
K = 20; % # iterations
nBlocks = 100;
Pt_dBm = 30;% Total transmit power in dBm scale
Pt = 10.^((Pt_dBm-30)/10);
No_dBm = -90; % Noise power in dBm scale
No = 10^((No_dBm-30)/10);
H_sd = 10; % Altitude
dDbar = 600; % horizontal BS-user distance
dpbar = 2; % vertical distance between IRS and projection of IRS to x-axis
dhbar = 2; % horizontal distance between user and projection of IRS to x-axis

dD = sqrt(dDbar^2 + H_sd^2); % Distance between BS-user direct link
dBI = sqrt((dDbar - dhbar)^2 + dpbar^2); % Distance between BS and IRS
dIU = sqrt(dhbar^2 + dpbar^2 + H_sd^2); % Distance between IRS and User
beta0_dB = -30;
beta0 =  10^(beta0_dB/10);
alphaD = 3.5; % Path loss exponent for BS-user direct link
alphaBI = 1.5; % Path loss exponent for BS-IRS link
alphaIU = 1.5; % Path loss exponent for IRS-user link
betaD = beta0/(dD^alphaD); % distance-dependent path loss for BS-user direct link
betaBI = beta0/(dBI^alphaBI); % distance-dependent path loss for BS-IRS link
betaIU = beta0/(dIU^alphaIU); % distance-dependent path loss for IRS-user link
rate_AO = zeros(length(num_elements),1); 
rate_random = zeros(length(num_elements),1); 
rate_withoutIRS = 0;

for blk = 1:(nBlocks)
    h_sd = sqrt(0.5*betaD)*(randn(Nt,1) + 1j*randn(Nt,1)); % Direct BS-user channel
    w=sqrt(Pt)*(h_sd)/norm(h_sd);
    gamma =  (abs(h_sd'*w))^2/No; % Receive SNR
    rate_withoutIRS = rate_withoutIRS + log2(1+gamma);
    for m=1:length(num_elements)
        N = num_elements(m);
        H_sr = sqrt(0.5*betaBI)*(randn(N,Nt) + 1j*randn(N,Nt)); % BS-IRS channel
        h_rd = sqrt(0.5*betaIU)*(randn(N,1) + 1j*randn(N,1)); %% IRS-user channel
        
        theta = exp(1i*2*pi.*rand(N,1));
        h_eff = h_rd'*diag(theta)*H_sr + h_sd';
        w=sqrt(Pt)*(h_eff)'/norm(h_eff);
        gamma =  abs(h_eff*w)^2/No; 
        rate_random(m) = rate_random(m) + log2(1+gamma);         

        Phi_0 = angle(h_sd'*w);        
        % Optimization
        for kk = 1:K
            % Calculation of theta
            theta = zeros(N,1);
            for n=1:N
                theta(n) = Phi_0+angle(h_rd(n))-angle(H_sr(n,:)*w);
            end
            Theta = diag(exp(1i*theta));
            h_eff = h_rd'*Theta*H_sr+h_sd';

            % Design of w using MRT
            w=sqrt(Pt)*(h_eff)'/norm(h_eff);
            Phi_0 = angle(h_sd'*w);
        end
        gamma = (abs(h_eff*w))^2/No; % Receive SNR
        rate_AO(m) = rate_AO(m) + log2(1+gamma);        
    end
end
rate_AO = rate_AO/nBlocks;
rate_random = rate_random/nBlocks;
rate_withoutIRS = rate_withoutIRS/nBlocks;

semilogy(num_elements,rate_AO,'r  s-','linewidth',3.0,'MarkerFaceColor','r','MarkerSize',9.0);
hold on; grid on;
semilogy(num_elements,rate_random,'b  s-','linewidth',3.0,'MarkerFaceColor','b','MarkerSize',9.0);
semilogy(num_elements,rate_withoutIRS*ones(length(num_elements),1),'k  o-','linewidth',3.0,'MarkerFaceColor','k','MarkerSize',9.0);
legend('AO','Random phase','Without IRS','Location','NorthWest')
xlabel('Number of reflecting elements N');
ylabel('Rate (bps/Hz)');
title('Rate vs number of reflecting elements for MISO-IRS system');