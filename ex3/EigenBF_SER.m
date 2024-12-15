close all; clear; clc


QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:14;
SNR=10.^(SNRdB./10);

SER=zeros(size(SNRdB));
N=100000; % Number of pairs per SNR point
N_Tx=4;


for k=1:length(SNRdB),
    s_hat=zeros(1,N); 
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(1,N,QPSK_vec);
    n=(randn(2,N)+1i*randn(2,N))/sqrt(2);
         
    for kk=1:N,
        % Signal generation
        H=(randn(2,N_Tx)+1i*randn(2,N_Tx))/sqrt(2); 
        
        [V,D]=eig(H'*H); 
        Precoder=V(:,end); 
        
        y=H*Precoder*s(kk)+rho*n(:,kk); 
        
        % Signal Detection
        s_hat(kk)=(H*Precoder)'*y/norm(H*Precoder)^2; 
    end; 
    
    s_tilde=sign(real(s_hat))/sqrt(2)+1i*sign(imag(s_hat))/sqrt(2);
    
    NumOfErrors=sum(abs(s_tilde-s)>eps); 
    SER(k)=NumOfErrors/N; 
end;

semilogy(SNRdB,SER); grid; 
xlabel('SNR(dB)'); 
ylabel('SER'); 
title(['EigenBF 2X',num2str(N_Tx),' Rayleigh']); 