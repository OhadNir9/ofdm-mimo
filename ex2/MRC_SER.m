close all; clear; clc

N_Rx=3; % Num of Rx antennas


QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:20;
SNR=10.^(SNRdB./10);

SER=zeros(size(SNRdB));
N=100000;
 


for k=1:length(SNRdB),
    s_hat=zeros(N,1); 
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(N,1,QPSK_vec);
    h=(randn(N_Rx,N)+1i*randn(N_Rx,N))/sqrt(2);
    n=(randn(N_Rx,N)+1i*randn(N_Rx,N))/sqrt(2);
    
    for kk=1:N, 
        y=h(:,kk)*s(kk)+rho*n(:,kk); 
        % Signal Detection  
        s_hat(kk)=h(:,kk)'*y/norm(h(:,kk))^2;
    end; 
    
    s_tilde=sign(real(s_hat))/sqrt(2)+1i*sign(imag(s_hat))/sqrt(2);
    
    NumOfErrors=sum(abs(s_tilde-s)>eps); 
    SER(k)=NumOfErrors/N; 
end;

semilogy(SNRdB,SER); grid; 
xlabel('SNR(dB)'); 
ylabel('SER'); 
title(['MRC Rayleigh ',num2str(N_Rx), ' Antennas']); 