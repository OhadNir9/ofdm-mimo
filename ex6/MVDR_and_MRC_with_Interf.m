close all; clear; clc

N_Rx=4; % Num of Rx antennas

SIRdB=5; 


QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:20;
SNR=10.^(SNRdB./10);

SER_MRC=zeros(size(SNRdB));
SER_MVDR=zeros(size(SNRdB));
N=100000;
 

InterfereGain=10^(-SIRdB/20); 

for k=1:length(SNRdB),
    s_hat_MRC=zeros(N,1); 
    s_hat_MVDR=zeros(N,1); 
    rho=1./sqrt(SNR(k));
    
    % Signal Generation
    s=randsrc(N,1,QPSK_vec);
    h=(randn(N_Rx,N)+1i*randn(N_Rx,N))/sqrt(2);
    g=(randn(N_Rx,N)+1i*randn(N_Rx,N))/sqrt(2); % Channel to interferer
    r=(randn(1,N)+1i*randn(1,N))/sqrt(2); % Interferer signal
    n=(randn(N_Rx,N)+1i*randn(N_Rx,N))/sqrt(2);
    
    for kk=1:N, 
        y=h(:,kk)*s(kk)+InterfereGain*g(:,kk)*r(kk)+rho*n(:,kk); 
        % MRC Signal Detection  
        s_hat_MRC(kk)=h(:,kk)'*y/norm(h(:,kk))^2;
        
        %MVDR Signal Detection
        C=InterfereGain^2*g(:,kk)*g(:,kk)'+rho^2*eye(N_Rx);
        InvC=inv(C); 
        s_hat_MVDR(kk)=h(:,kk)'*InvC*y/(h(:,kk)'*InvC*h(:,kk));
    end; 
    
    s_tilde_MRC=sign(real(s_hat_MRC))/sqrt(2)+1i*sign(imag(s_hat_MRC))/sqrt(2);
    s_tilde_MVDR=sign(real(s_hat_MVDR))/sqrt(2)+1i*sign(imag(s_hat_MVDR))/sqrt(2);
    NumOfErrors_MRC=sum(abs(s_tilde_MRC-s)>eps); 
    NumOfErrors_MVDR=sum(abs(s_tilde_MVDR-s)>eps); 
    SER_MRC(k)=NumOfErrors_MRC/N; 
    SER_MVDR(k)=NumOfErrors_MVDR/N;
end;

semilogy(SNRdB,SER_MRC); grid;
hold on
semilogy(SNRdB,SER_MVDR,'r'); 
xlabel('SNR(dB)'); 
ylabel('SER'); 
legend('MRC','MVDR')
title(['MRC and MVDR Rayleigh ',num2str(N_Rx), ' Antennas at SINR=',num2str(SIRdB),'dB']); 