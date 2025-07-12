import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
from scipy.special import comb
import MIMO

blockLength = 1000; # number of symbols per block
nBlocks = 10000; # number of blocks
r = 2; # Number of receive antennas
t = 2; # number of transmit antennas
EbdB = np.arange(1.0,33.1,4.0); # Energy per bit in dB
Eb = 10**(EbdB/10); # Energy per bit
No = 1; # Complex noise power No
Es = 2*Eb; # Energy per symbol Eb
SNR = Es/No; # Signal-to-noise power ratio SNR
SNRdB = 10*np.log10(SNR); # SNR in dB
BER_ZF = np.zeros(len(EbdB)); # Bit error rate for ZF receiver
BER_LMMSE = np.zeros(len(EbdB)); # Bit error rate for LMMSE receiver
BERt = np.zeros(len(EbdB)); # Bit error values from formula


for blk in range(nBlocks):   
    # MIMO channel matrix
    H = (nr.normal(0.0, 1.0,(r,t))+1j*nr.normal(0.0, 1.0,(r,t)))/np.sqrt(2);
    # Complex Gaussian noise of power No
    noise = nr.normal(0.0, np.sqrt(No/2), (r,blockLength))+ \
    1j*nr.normal(0.0, np.sqrt(No/2), (r,blockLength));
    BitsI = nr.randint(2,size=(t,blockLength)); # Bits for I channel
    BitsQ = nr.randint(2,size=(t,blockLength)); # Bits for Q channel
    Sym = (2*BitsI-1)+1j*(2*BitsQ-1); # Complex QPSK symbols

    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K])*Sym; # Tx symbols after power scaling
        RxSym = np.matmul(H,TxSym) + noise; # Rx symbols in AWGN
        
        # ZF Receiver
        ZFRx = nl.pinv(H); #ZF receiver
        ZFout = np.matmul(ZFRx, RxSym); #output of ZF receiver
        DecBitsI_ZF = (np.real(ZFout)>0); #Decoding bits for I channel
        DecBitsQ_ZF = (np.imag(ZFout)>0); #Decoding bits for Q channel
        # Evaluating total number of bit errors for ZF receiver
        BER_ZF[K] = BER_ZF[K] + np.sum(DecBitsI_ZF != BitsI) + np.sum(DecBitsQ_ZF != BitsQ);
        
        # LMMSE Receiver
        LMMSERx = np.matmul(nl.inv(MIMO.AHA(H)+No*np.identity(t)/Es[K]),MIMO.H(H));
        LMMSEout = np.matmul(LMMSERx,RxSym); #Outputof LMMSE receiver
        DecBitsI_LMMSE = (np.real(LMMSEout)>0); #Decoding bits for I channel
        DecBitsQ_LMMSE = (np.imag(LMMSEout)>0); #Decoding bits for Q channel
        # Evaluating total number of bit errors for LMMSE receiver
        BER_LMMSE[K] = BER_LMMSE[K] + np.sum(DecBitsI_LMMSE != BitsI) \
            + np.sum(DecBitsQ_LMMSE != BitsQ);


BER_ZF = BER_ZF/blockLength/nBlocks/2/t; # Average BER for ZF Receiver
BER_LMMSE = BER_LMMSE/blockLength/nBlocks/2/t; # Average BER for LMMSE Receiver
L=r-t+1;    BERt = comb(2*L-1, L)/2**L/SNR**L; # BER for ZF from formulat

# Plot BER of ZF and LMMSE Receivers
plt.yscale('log')
plt.plot(SNRdB, BER_ZF,'g-');
plt.plot(SNRdB, BER_LMMSE,'b-.s');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for MIMO Channel')
plt.legend(["ZF","LMMSE", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 

