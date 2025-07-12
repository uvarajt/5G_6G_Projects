clc; clear all; close all;

M = 32; N = 16;
F_M = 1/sqrt(M)*dftmtx(M);
F_N = 1/sqrt(N)*dftmtx(N);
Ptx = eye(M);

delta_f = 15e3; %15kHz
T=1/delta_f;

X_DD = zeros(M,N);
X_DD(2,2) = 1;
X_TF = F_M*X_DD*F_N';
S = Ptx*F_M'*X_TF;
s = reshape(S,M*N,1);

figure()
bar3(X_DD);
axis tight;
xlabel('Doppler'); 
ylabel('Delay');
title('Basis function in DD-domain');

figure()
surf(real(X_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Basis function in TF-domain (Real)');

figure()
surf(imag(X_TF));
axis tight;
xlabel('Time'); 
ylabel('Subcarrier');
title('Basis function in TF-domain (Imag)');

figure()
plot((0:length(s)-1)*T/M,real(s));
axis tight;
ylim([-0.5 0.5]);
xlabel('Time');
title('Basis function in time-domain (Real)');

figure()
plot((0:length(s)-1)*T/M,imag(s));
axis tight;
ylim([-0.5 0.5]);
xlabel('Time');
title('Basis function in time-domain (Imag)');

