clc; clear all; close all;
Nr = 4; Nt = 4;
M = 2:2:40; % Number of IRS elements
K = 25; % Number of optimization iterations
nBlocks = 40;
Pt_dBm = 30;% Total transmit power in dBm scale
Pt = 10.^((Pt_dBm-30)/10);
No_dBm = -90; % Noise power in dBm scale
No = 10^((No_dBm-30)/10);
H = 10; % Altitude
dDbar = 600; % Horizontal BS-user distance
dpbar = 2; % Horizontal distance between IRS and projection of IRS to x-axis
dhbar = 2; % Horizontal distance between user and projection of IRS to x-axis

dD = sqrt(dDbar^2 + H^2); % Distance between BS-user direct link
dBI = sqrt((dDbar - dhbar)^2 + dpbar^2); % Distance between BS and IRS
dIU = sqrt(dhbar^2 + dpbar^2 + H^2); % Distance between IRS and BS
beta0_dB = -30;
beta0 =  10^(beta0_dB/10);
alphaD = 3.5; % Path loss exponent for BS-user direct link
alphaBI = 1.5; % Path loss exponent for BS-IRS link
alphaIU = 1.5; % Path loss exponent for IRS-user link
betaD = beta0/(dD^alphaD); % Distance-dependent path loss for BS-user direct link
betaBI = beta0/(dBI^alphaBI); % Distance-dependent path loss for BS-IRS link
betaIU = beta0/(dIU^alphaIU); % Distance-dependent path loss for IRS-user link
CAP_AO = zeros(length(M),1); CAP_random = zeros(length(M),1); CAP_withoutIRS = 0;
for blk = 1:(nBlocks)
    blk
    for m =1:length(M)
        H = sqrt(0.5*betaD)*(randn(Nr,Nt) + 1j*randn(Nr,Nt)); % Direct BS-user channel
        T = sqrt(0.5*betaBI)*(randn(M(m),Nt) + 1j*randn(M(m),Nt)); % BS-IRS channel
        R = sqrt(0.5*betaIU)*(randn(Nr,M(m)) + 1j*randn(Nr,M(m))); % IRS-user channel        
        % Alternating optimization
        % Initialization of alpha and Q
        alpha = ones(M(m),1);
        Q = (Pt/Nt)*eye(Nt);
        Q_sqrt=sqrt(Pt/Nt)*eye(Nt);
        H_dash = H*Q_sqrt;
        T_dash = T*Q_sqrt;
        
        for kk = 1:K
            % Calculation of alpha
            alpha = OPT_REFL_COEFF(H_dash,R,T_dash,alpha,No,Nr,M(m));
            % Calculation of Q
            H_tilde = H+R*diag(alpha)*T;
            [Q,Q_sqrt,CAP] = OPT_Q_MIMO(H_tilde,Pt,No);
         
            H_dash = H*Q_sqrt;
            T_dash = T*Q_sqrt;
        end
        CAP_AO(m) = CAP_AO(m)+ CAP;
        
        % Random IRS phase shifts
        alpha = exp(1i*2*pi.*rand(M(m),1));
        % Calculation of Q
        H_tilde = H + R*diag(alpha)*T;
        [~,~,CAP] = OPT_Q_MIMO(H_tilde,Pt,No);
        CAP_random(m) = CAP_random(m) + CAP;
        
    end
    % Without IRS
    [~,~,CAP] = OPT_Q_MIMO(H,Pt,No);
    CAP_withoutIRS = CAP_withoutIRS + CAP;
end
CAP_AO = CAP_AO/nBlocks;
CAP_random = CAP_random/nBlocks;
CAP_withoutIRS = CAP_withoutIRS/nBlocks;

semilogy(M,real(CAP_AO),'r  s-','linewidth',3.0,'MarkerFaceColor','r','MarkerSize',9.0);
hold on;
semilogy(M,real(CAP_random),'b  s-','linewidth',3.0,'MarkerFaceColor','b','MarkerSize',9.0);
semilogy(M,real(CAP_withoutIRS)*ones(length(M),1),'k  o-','linewidth',3.0,'MarkerFaceColor','k','MarkerSize',9.0);
grid on;
legend('AO','Random phase','Without IRS','Location','NorthWest')
xlabel('Number of reflecting elements M');
ylabel('Rate (bps/Hz)');
title('Rate vs number of reflecting elements for MIMO-IRS system');