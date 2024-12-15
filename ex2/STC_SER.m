close all; clear; clc


QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:20;
SNR=10.^(SNRdB./10);

SER=zeros(size(SNRdB));
N=100000; % Number of pairs per SNR point
 


for k=1:length(SNRdB),
    s_hat=zeros(2,N); 
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(2,N,QPSK_vec);
    h=(randn(2,N)+1i*randn(2,N))/sqrt(2);
    n=(randn(2,N)+1i*randn(2,N))/sqrt(2);
         
    for kk=1:N,
        % Signal generation
        H_stc=1/sqrt(2)*[h(1,kk) h(2,kk); h(2,kk)' -h(1,kk)']; 
        y=H_stc*s(:,kk)+rho*n(:,kk); 
        
        % Signal detection
        s_hat(1,kk)=H_stc(:,1)'*y/norm(H_stc(:,1))^2; 
        s_hat(2,kk)=H_stc(:,2)'*y/norm(H_stc(:,2))^2; 
    end; 
    
    s_tilde=sign(real(s_hat))/sqrt(2)+1i*sign(imag(s_hat))/sqrt(2);
    
    NumOfErrors=sum(abs(s_tilde(:)-s(:))>eps); 
    SER(k)=NumOfErrors/2/N; 
end;

semilogy(SNRdB,SER); grid; 
xlabel('SNR(dB)'); 
ylabel('SER'); 
title('STC 2X1 Rayleigh'); 