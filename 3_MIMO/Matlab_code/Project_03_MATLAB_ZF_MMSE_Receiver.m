close all;
clear all;

blockLength = 1000;
nBlocks=10000;
r=2;
t=2;
EbdB = 1.0:4.0:33.0;
Eb = 10.^(EbdB/10);
No = 1;
Es = 2*Eb;
SNR = Es/No;
SNRdB = 10*log10(SNR);
BER_ZF = zeros(1,length(EbdB));
BER_LMMSE = zeros(1,length(EbdB));
BERt = zeros(1,length(EbdB));

for blk = 1:nBlocks
    H = 1/sqrt(2)*(randn(r,t) + 1j*randn(r,t));
    BitsI = randi([0,1],t,blockLength);
    BitsQ = randi([0,1],t,blockLength);
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1);
    noise = sqrt(No/2)*(randn(r,blockLength)+1j*randn(r,blockLength));
    for K = 1:length(EbdB)
        TxSym = sqrt(Eb(K))*Sym;
        RxSym = H*TxSym + noise;
        
        ZFRx = pinv(H);
        ZFout = ZFRx*RxSym;
        DecBitsI_ZF = (real(ZFout)>0);
        DecBitsQ_ZF = (imag(ZFout)>0);
        BER_ZF(K) = BER_ZF(K) + sum(sum(DecBitsI_ZF ~= BitsI)) + sum(sum(DecBitsQ_ZF ~= BitsQ));

        LMMSERx = inv(H'*H+No*eye(t)/Es(K))*H';
        LMMSEout = LMMSERx*RxSym;
        DecBitsI_LMMSE = (real(LMMSEout)>0);
        DecBitsQ_LMMSE = (imag(LMMSEout)>0);
        BER_LMMSE(K) = BER_LMMSE(K) + sum(sum(DecBitsI_LMMSE ~= BitsI)) + sum(sum(DecBitsQ_LMMSE ~= BitsQ));
        
    end
end
BER_ZF = BER_ZF/blockLength/nBlocks/2/t;
BER_LMMSE = BER_LMMSE/blockLength/nBlocks/2/t;
L=r-t+1;    
BERt = nchoosek(2*L-1, L)/2^L./SNR.^L;


semilogy(SNRdB,BER_ZF,'g  s','linewidth',3.0,'MarkerFaceColor','g','MarkerSize',9.0);
hold on;
semilogy(SNRdB,BERt,'g -.','linewidth',3.0,'MarkerFaceColor','g','MarkerSize',9.0);
semilogy(SNRdB,BER_LMMSE,'r -- o','linewidth',3.0,'MarkerFaceColor','r','MarkerSize',9.0);
axis tight;
grid on;
legend('ZF','ZF Theory','LMMSE')
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR(dB) for MIMO ZF/ MMSE Receivers');



