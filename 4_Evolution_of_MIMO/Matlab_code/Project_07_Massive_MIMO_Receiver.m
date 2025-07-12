clear all;
close all;
clc;

ITER = 5000;
K = 10; % Number of users
Mv = 20:30:500; % Number of BS antennas
Eu_dB = 10;  Eu = 10^(Eu_dB/10);
rate_MRC = zeros(1,length(Mv)) ;
bound_MRC = zeros(1,length(Mv));
rate_ZF = zeros(1,length(Mv));


for it=1:ITER
    D = Dmatrix(K); beta = diag(D);
    for mx=1:length(Mv)
        M = Mv(mx);
        %pu = Eu; % No power scaling
        pu = Eu/M; % Power scaling
        H = sqrt(1/2)*(randn(M,K)+1i*randn(M,K));
        G = H*sqrt(D);
        g1 = G(:,1); % clumn of user k
        nr_MRC = pu*norm(g1)^2;
        nr_bound_MRC = pu*M*beta(1);
        dr_MRC = 1 + pu*norm(g1'/norm(g1)*G(:,2:K))^2;
        dr_bound_MRC = 1 + pu*sum(beta(2:K));
        rate_MRC(mx) = rate_MRC(mx)+log2(1+nr_MRC/dr_MRC);
        bound_MRC(mx) = bound_MRC(mx)+log2(1+nr_bound_MRC/dr_bound_MRC);

        nr_ZF = pu; invGG = inv(G'*G);
        dr_ZF = invGG(1,1);
        rate_ZF(mx) = rate_ZF(mx)+log2(1+nr_ZF/dr_ZF);
        
    end
end
rate_MRC = rate_MRC/ITER;
bound_MRC = bound_MRC/ITER;
rate_ZF = rate_ZF/ITER;

figure;
plot(Mv,rate_MRC,'b o-','LineWidth',3,'MarkerFaceColor','blue','MarkerSize',9.0)
hold on
plot(Mv,bound_MRC,'g -.','LineWidth',3)
plot(Mv,rate_ZF,'r -s','LineWidth',3,'MarkerFaceColor','red','MarkerSize',9.0)
grid on
title('Information Rate of Massive MIMO System with MRC, ZF Receivers')
legend('MRC','Bound MRC','ZF','Location','SouthEast');
xlabel('Number of BS Antennas')
ylabel('Uplink Sum Rate (bits/s/Hz)')
