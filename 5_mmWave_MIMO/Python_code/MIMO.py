import numpy as np
import numpy.linalg as nl
import numpy.random as nr
from scipy.stats import norm
from scipy.stats import unitary_group
import pdb


def Dmatrix(K):
    var_nr = (10**(8/10))**2; mean_nr = 3;
    mu_nr = np.log10(mean_nr**2/np.sqrt(var_nr+mean_nr**2)); 
    sigma_nr = np.sqrt(np.log10(var_nr/(mean_nr**2+1)));
    nr = np.random.lognormal(mu_nr,sigma_nr,K);
    dr = np.random.randint(100,1000,K)/100;
    beta = nr/dr**3.0;
    return beta;

def DFTmat(K):
    kx, lx = np.meshgrid(np.arange(K), np.arange(K))
    omega = np.exp(-2*np.pi*1j/K)
    dftmtx = np.power(omega,kx*lx)
    return dftmtx

def Q(x):
    return 1-norm.cdf(x);

def QPSK(m,n):
    return ((2*nr.randint(2,size=(m,n))-1)+1j*(2*nr.randint(2,size=(m,n))-1))/np.sqrt(2);

def H(G):
    return np.conj(np.transpose(G));

def ArrayDictionary(G,t):
    lxx = 2/G*np.arange(G)-1;
    lx, kx = np.meshgrid(lxx, np.arange(t))
    omega = np.exp(-1j*np.pi)
    dmtx = 1/np.sqrt(t)*np.power(omega,kx*lx)
    return dmtx

def RF_BB_matrices(numAnt,numRF,N_Beam):
    NBlk = numAnt/numRF;
    RFmat = 1/np.sqrt(numAnt)*DFTmat(numAnt);
    U = unitary_group.rvs(numRF);
    V = unitary_group.rvs(int(N_Beam/NBlk));
    CenterMat = np.concatenate((np.identity(int(N_Beam/NBlk)), 
                                np.zeros((int(numRF-N_Beam/NBlk),int(N_Beam/NBlk)))),axis=0);
    BB_diag = nl.multi_dot([U,CenterMat,H(V)]);
    BBmat = np.kron(np.identity(int(NBlk)),BB_diag); 
    return RFmat, BBmat

def OMP(y,Q,thrld):
    [rq,cq] = Q.shape; 
    set_I = np.zeros(cq);  
    r_prev = np.zeros((rq,1)) + 1j*np.zeros((rq,1)); 
    hb_omp = np.zeros((cq,1)) + 1j*np.zeros((cq,1));
    r_curr = y; 
    Qa = np.zeros((rq,cq)) + 1j*np.zeros((rq,cq)); 
    ix1 = 0;
    while np.absolute(nl.norm(r_prev)**2 - nl.norm(r_curr)**2) > thrld:
        m_ind = np.argmax(np.absolute(np.matmul(H(Q),r_curr))); 
        set_I[ix1] = m_ind;
        Qa[:,ix1] = Q[:,m_ind];
        hb_ls = np.matmul(nl.pinv(Qa[:,0:ix1+1]),y); 
        r_prev = r_curr;
        r_curr = y - np.matmul(Qa[:,0:ix1+1],hb_ls); 
        ix1 = ix1 + 1;

    set_I_nz = set_I[0:ix1];
    hb_omp[set_I_nz.astype(int)] = hb_ls;
    return hb_omp

def sparse_h_creation(M_tau,N_nu,chan_coef,taps,delay_taps,Doppler_taps):
    h_sparse = np.zeros([M_tau*N_nu,1])+1j*np.zeros([M_tau*N_nu,1]);
    sparse_loc = np.zeros(taps);
    for i in range(taps):
        sparse_loc[i] = N_nu*delay_taps[i] + Doppler_taps[i];
    h_sparse[sparse_loc.astype(int)] = np.reshape(chan_coef,[chan_coef.size,1]);
    return h_sparse


def SOMP(Opt, Dict, Ryy, numRF):
    rq, cq = np.shape(Dict); 
    Res = Opt; 
    RF = np.zeros((rq,numRF))+1j*np.zeros((rq,numRF));   
    for iter1 in range(numRF):
        phi = nl.multi_dot([H(Dict),Ryy,Res]); 
        phi_phiH = AAH(phi); 
        m_ind = np.argmax(np.abs(np.diag(phi_phiH)));
        RF[:,iter1] = Dict[:,m_ind];  
        RFc = RF[:,0:iter1+1];
        BB = nl.multi_dot([nl.inv(nl.multi_dot([H(RFc),Ryy,RFc])),H(RFc),Ryy,Opt]);
        Res = (Opt-np.matmul(RFc,BB))/nl.norm(Opt-np.matmul(RFc,BB));    
    return  BB, RF


def SOMP_Est(y,Qbar,thrld):
    rq,cq = np.shape(Qbar);
    ry,cy = np.shape(y);
    set_I = np.zeros((cq,1));
    r_prev = np.zeros((ry,cy))+1j*np.zeros((ry,cy));
    hb_OMP = np.zeros((cq,cy))+1j*np.zeros((cq,cy));
    r_curr = y; 
    Q_a = np.zeros((rq,cq))+1j*np.zeros((rq,cq)); 
    ix1 = 0;
    while(abs((nl.norm(r_prev,2))**2 - (nl.norm(r_curr,2))**2) > thrld):
        psi = nl.multi_dot([H(Qbar),r_curr]);
        m_ind = np.argmax(np.abs(np.diag(AAH(psi))));
        set_I[ix1] = m_ind;
        Q_a[:,ix1] = Qbar[:,m_ind];
        Q_c = Q_a[:,0:ix1+1];
        Hb_LS = np.matmul(nl.pinv(Q_c),y);
        r_prev = r_curr;
        r_curr = y - np.matmul(Q_c,Hb_LS);
        ix1 = ix1 + 1;
    set_I_nz = set_I[0:ix1];    
    hb_OMP[set_I_nz.astype(int).flatten(),:] = Hb_LS;
    return hb_OMP


def MSE_time_domain(H,Ht,Fsub,r,t,Nt):
    Ht_est = np.zeros((r,t,Nt))+1j*np.zeros((r,t,Nt));
    for tx in range(t):
        for rx in range(r):
            Ht_est[rx,tx,:] = np.matmul(nl.pinv(Fsub),H[rx,tx,:]);
    MSE_td = nl.norm(Ht.flatten()-Ht_est.flatten())**2/t/r/Nt;
    return MSE_td

def SBL(y,Q,sigma_2):
    N, M = np.shape(Q);
    Gamma = np.identity(M);
    for iter in range(50):
        Sigma = nl.inv(1/sigma_2*np.matmul(H(Q),Q) + nl.inv(Gamma));
        mu = 1/sigma_2*nl.multi_dot([Sigma,H(Q),y]);
        Gamma = np.diag(np.diag(Sigma)+np.abs(mu).flatten()**2);
    return mu, Gamma 
        


def mmWaveMIMOChannelGenerator(A_R,A_T,G,L):
    t = A_T.shape[0];
    r = A_R.shape[0];
    Psi = np.zeros(shape=(t*r,L))+np.zeros(shape=(t*r,L))*1j;
    tax = nr.choice(G, L, replace=False);
    rax = nr.choice(G, L, replace=False);
    alpha = 1/np.sqrt(2)*(nr.normal(0,1,L)+1j*nr.normal(0,1,L));
    A_T_genie = A_T[:, tax];
    A_R_genie = A_R[:, rax];
    for jx in range(L):
        Psi[:,jx] = np.kron(np.conj(A_T[:,tax[jx]]),A_R[:,rax[jx]]);       
    return alpha, Psi, A_R_genie, A_T_genie


def mmWaveMIMO_OFDMChannelGenerator(A_R,A_T,L,numTaps):
    t,G = np.shape(A_T);
    r,G = np.shape(A_R);
    Ht = np.zeros((r,t,numTaps)) + 1j*np.zeros((r,t,numTaps));
    Psi = np.zeros(shape=(t*r,L))+np.zeros(shape=(t*r,L))*1j;
    tax = nr.choice(G, L, replace=False);
    rax = nr.choice(G, L, replace=False);
    A_T_genie = A_T[:, tax];
    A_R_genie = A_R[:, rax];
    for jx in range(L):
        Psi[:,jx] = np.kron(np.conj(A_T[:,tax[jx]]),A_R[:,rax[jx]]);
    for px in range(numTaps):
        alpha = 1/np.sqrt(2)*(nr.normal(0,1,L)+1j*nr.normal(0,1,L));
        Ht[:,:,px] = np.sqrt(t*r/L)*nl.multi_dot([A_R_genie,np.diag(alpha),H(A_T_genie)])
    return Ht, Psi, A_R_genie, A_T_genie



def AHA(A):
    return np.matmul(H(A),A)

def AAH(A):
    return np.matmul(A,H(A))

def mimo_capacity(Hmat, TXcov, Ncov):
    r, c = np.shape(Hmat);
    inLD = np.identity(r) + nl.multi_dot([nl.inv(Ncov),Hmat,TXcov,H(Hmat)]);
    C = np.log2(nl.det(inLD));
    return np.abs(C)


def OPT_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = 0;
    while not CAP:
        onebylam = (SNR + sum(1/S[0:t]**2))/t;
        if  onebylam - 1/S[t-1]**2 >= 0:
            optP = onebylam - 1/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP));
        elif onebylam - 1/S[t-1]**2 < 0:
            t = t-1;            
    return CAP

def EQ_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = sum(np.log2(1+ S[0:t]**2 * SNR/t));
    return CAP


def MPAM_DECODER(EqSym,M):
    DecSym = np.round((EqSym+M-1)/2);
    DecSym[np.where(DecSym<0)] = 0;
    DecSym[np.where(DecSym>(M-1))] = M-1      
    return DecSym

def MQAM_DECODER(EqSym,M):
    sqM = np.int(np.sqrt(M));
    DecSym = np.round((EqSym+sqM-1)/2);
    DecSym[np.where(DecSym<0)]=0;
    DecSym[np.where(DecSym>(sqM-1))]=sqM-1      
    return DecSym

def PHYDAS(L_f,N):
    H1=0.971960;
    H2=np.sqrt(2)/2;
    H3=0.235147;
    fh=1+2*(H1+H2+H3);
    hef=np.zeros((1,L_f+1));
    for i in range(L_f+1):
        hef[0,i]=1-2*H1*np.cos(np.pi*i/(2*N))+2*H2*np.cos(np.pi*i/N)-2*H3*np.cos(np.pi*i*3/(2*N));

    hef = hef/fh;
    p_k = hef/nl.norm(hef);
    return(p_k)

def UPSAMPLE(H,k):
    m = H.shape[0];
    n = H.shape[1];
    G = np.zeros((int(m*k),n))+1j*np.zeros((int(m*k),n));
    for ix in range(m):
        G[ix*k,:] = H[ix,:];
        
    return(G)


def DOWNSAMPLE(H,k):
    m = H.shape[0];
    n = H.shape[1];
    G = np.zeros((int(m/k),n))+1j*np.zeros((int(m/k),n));
    for ix in range(int(m/k)):
        G[ix,:] = H[ix*k,:];
        
    return(G)

def H_eff_creation(M,N,taps,chan_coef,delay_taps,Doppler_taps,Prx,Ptx):
    H_mat = np.zeros([M*N,M*N]);
    F_N = 1/np.sqrt(N)*DFTmat(N);
    omega = np.exp(1j*2*np.pi/(M*N));
    for tx in range(taps):
        DiagMat = np.diag(omega**(np.arange(0,M*N)*Doppler_taps[tx]));
        CircMat = np.roll(np.identity(M*N),-int(delay_taps[tx]),axis=1);
        H_mat = H_mat + chan_coef[tx]*np.matmul(CircMat,DiagMat);

    H_eff = nl.multi_dot([np.kron(F_N,Prx),H_mat,np.kron(H(F_N),Ptx)]);
    return(H_eff)

def Pilot_impulse_CE(M_tau,N_nu,thrld,Y_DD,pp_dB):
    h_est = np.array([]);
    delay_taps_est = np.array([]);
    Doppler_taps_est = np.array([]);
    num_est_taps = 0;
    for m_i in range(M_tau):
        for n_i in range(N_nu):
            if np.abs(Y_DD[m_i,n_i]) > thrld:
                h_est = np.concatenate((h_est,[Y_DD[m_i,n_i]/np.sqrt(10**(pp_dB/10))]));
                delay_taps_est =  np.concatenate((delay_taps_est, [m_i]));
                Doppler_taps_est = np.concatenate((Doppler_taps_est,[n_i]))
                num_est_taps = num_est_taps + 1;
    
         
    return h_est, delay_taps_est, Doppler_taps_est, num_est_taps


def Dict_mtx_creation(M,M_tau,N,N_nu,N_p,pilots):
    del_li_diag = np.zeros([N_p,1])+1j*np.zeros([N_p,1]);
    iden = np.identity(N_p);
    Omega_i_j = np.zeros([N_p,N_p,M_tau*N_nu])+1j*np.zeros([N_p,N_p,M_tau*N_nu]);
    pilotsCol = np.reshape(pilots,[pilots.size,1]);
    Omega = np.zeros([N_p,M_tau*N_nu])+1j*np.zeros([N_p,M_tau*N_nu]);
    for ix in range(M_tau):
        if ix==0:   
            del_li_diag = np.exp(1j*2*np.pi/(M*N)*np.arange(0,N_p));
        else:
            del_li_diag[0:N_p-ix] = np.exp(1j*2*np.pi/(M*N)*np.arange(0,N_p - ix));
            del_li_diag[N_p-ix:] = np.exp(1j*2*np.pi/(M*N)*np.arange(-ix,0));
        for jx in range(N_nu):
            CircMat = np.roll(iden,-ix,axis=1)
            DiagMat = np.diag(del_li_diag**jx);
            omega = np.matmul(CircMat,DiagMat);
            Omega_i_j[:,:,ix*N_nu+jx] = omega;


    for ix in range(M_tau*N_nu):
        Omega[:,ix:ix+1] = np.matmul(Omega_i_j[:,:,ix],pilotsCol);
        
    return(Omega)


def OPT_REFL_COEFF(H_dash, R, T_dash, alpha, No, Nr, Nt, M):
    for m in range(M):
        c= np.zeros(H_dash.shape);
        for m1 in range(M):
            if (m1 != m):
                c = c + alpha[m1]*nl.multi_dot([R[:,m1:m1+1],T_dash[m1:m1+1,:]]);
        A = np.identity(Nr) + AAH(H_dash + c)/No + AAH(np.matmul(R[:,m:m+1],T_dash[m:m+1,:]))/No;
        B = nl.multi_dot([R[:,m:m+1],T_dash[m:m+1,:],np.conj(H_dash.T + c.T)])/No;
       
        if (np.trace(np.matmul(nl.pinv(A),B))!=0):
           eval, evec = nl.eig(np.matmul(nl.pinv(A),B));
           eindex = np.argmax(np.abs(eval))
           alpha[m] = np.exp(-1j*np.angle(eval[eindex]));
        else:
           alpha[m] = 1;
        
    return alpha

def OPT_Q_MIMO(Heff,Pt,No):
    U, S, V = nl.svd(Heff,full_matrices=False)
    t = len(S);
    CAP = 0;
    while not CAP:
        onebylam = (Pt + sum(No/S[0:t]**2))/t;
        if  onebylam - No/S[t-1]**2 >= 0:
            optP = onebylam - No/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP/No));
        elif onebylam - No/S[t-1]**2 < 0:
            t = t-1;            
    Q = nl.multi_dot([V[:,0:t],np.diag(optP),H(V[:,0:t])]);
    Q_sqrt = np.matmul(V[:,0:t],np.diag(np.sqrt(optP)));
    
    return Q, Q_sqrt, CAP

    

