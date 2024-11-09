close all; clear; clc

QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:30;
SNR=10.^(SNRdB./10);

SER=zeros(size(SNRdB));
N=500000;

for k=1:length(SNRdB),
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(N,1,QPSK_vec);
    h=(randn(N,1)+1i*randn(N,1))/sqrt(2);
    n=(randn(N,1)+1i*randn(N,1))/sqrt(2);
    y=h.*s+rho*n;
    
    % Signal Detection    
    s_hat=y./h;
    s_tilde=sign(real(s_hat))/sqrt(2)+1i*sign(imag(s_hat))/sqrt(2);
    
    NumOfErrors=sum(abs(s_tilde-s)>eps); 
    SER(k)=NumOfErrors/N; 
end;

semilogy(SNRdB,SER); grid; 
xlabel('SNR(dB)'); 
ylabel('SER'); 
title('SISO Rayleigh'); 