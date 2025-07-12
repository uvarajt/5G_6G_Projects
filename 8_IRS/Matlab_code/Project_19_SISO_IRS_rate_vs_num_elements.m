clc; clear; close all;

% Initial Parameter declaration
num_elements=2:2:40;       % number of IRS elements
nBlocks = 1000;
Pt_dBm = 30;               % Total transmit power in dBm scale
Pt = 10.^((Pt_dBm-30)/10); % Transmit power in linear scale
No_dBm = -90;              % Noise power in dBm scale
No = 10^((No_dBm-30)/10);  % Noise power in linear scale
H = 10;                    % Altitude
dDbar = 600;               % horizontal BS-user distance
dpbar = 2;                 % vertical distance between IRS and projection of IRS to x-axis
dhbar = 2;                 % horizontal distance between user and projection of IRS to x-axis

dD = sqrt(dDbar^2 + H^2);                % Distance between BS-user direct link
dBI = sqrt((dDbar - dhbar)^2 + dpbar^2); % Distance between BS and IRS
dIU = sqrt(dhbar^2 + dpbar^2 + H^2);     % Distance between IRS and BS
beta0_dB = -30;
beta0 =  10^(beta0_dB/10);
alphaD = 3.5;                            % Path loss exponent for BS-user direct link
alphaBI = 1.5;                             % Path loss exponent for BS-IRS link
alphaIU = 1.5;                             % Path loss exponent for IRS-user link
betaD = beta0/(dD^alphaD);               % distance-dependent path loss for BS-user direct link
betaBI = beta0/(dBI^alphaBI);            % distance-dependent path loss for BS-IRS link
betaIU = beta0/(dIU^alphaIU);            % distance-dependent path loss for IRS-user link

rate_Random = zeros(1,length(num_elements));
rate_Opt = zeros(1,length(num_elements));
rate_withoutIRS = zeros(1,length(num_elements));

for blk=1:nBlocks
    h_sd =sqrt(0.5*betaD)*(randn(1,1)+1j*randn(1,1)); % Direct source-destination channel
    for m = 1:length(num_elements)
        N = num_elements(m);
        SNR=Pt/No;
        h_rd = sqrt(0.5*betaIU)*(randn(N,1)+1j*randn(N,1));  % Direct IRS-destination channel
        h_sr = sqrt(0.5*betaBI)*(randn(N,1)+1j*randn(N,1));  % Direct IRS-source channel
        r1=0;  r2=0;
        for n=1:N
            Theta_rand = rand*2*pi; % Case 1 Random Phase
            Theta_opt = mod(-angle(h_sd)-angle(h_sr(n))+angle(h_rd(n)),2*pi); % Case 2 Optimal Phase
            r1=r1+ conj(h_rd(n))*h_sr(n)*exp(1i*Theta_rand)+ conj(h_sd);
            r2=r2+ conj(h_rd(n))*h_sr(n)*exp(1i*Theta_opt)+ conj(h_sd);
        end
        rate_Random(m)=rate_Random(m)+log2(1+SNR*((abs(r1))^2));
        rate_Opt(m)=rate_Opt(m)+log2(1+SNR*((abs(r2))^2));     
    end
    rate_withoutIRS=rate_withoutIRS+log2(1+SNR*((abs(h_sd))^2));
end
rate_Random=rate_Random/nBlocks;
rate_Opt=rate_Opt/nBlocks;
rate_withoutIRS=rate_withoutIRS/nBlocks;

semilogy(num_elements,rate_Random,'r-*','linewidth',3.0,'MarkerFaceColor','r','MarkerSize',9.0);
hold on; grid on;
semilogy(num_elements,rate_Opt,'b-p','linewidth',3.0,'MarkerFaceColor','b','MarkerSize',9.0);
semilogy(num_elements,rate_withoutIRS,'g-s','linewidth',3.0,'MarkerFaceColor','g','MarkerSize',9.0);
xlabel('Number of elemeents'); ylabel('Rate (bps/Hz)');
title('Rate vs number of elements');
legend('Random phase shift','Optimal phase shift','without IRS','Location','best')