close all; clear; clc

N_rx=2;

QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);

if N_rx<3
    SNRdB=0:2:42;
else
    SNRdB=0:2:30;
end;
SNR=10.^(SNRdB./10);

SER_ZF=zeros(2,length(SNRdB));
SER_ML=zeros(2,length(SNRdB));
N=50000; % Number of pairs per SNR point

% Create a Matrix of 16 combinations for ML
CombinationLoc=zeros(2,16);
CombinationLoc(1,:)=fix([0:15]/4)+1;
CombinationLoc(2,:)=mod([0:15],4)+1;

CombinationMat=QPSK_vec(CombinationLoc);


for k=1:length(SNRdB),
    s_hat_ZF=zeros(2,N);
    s_tilde_ML=zeros(2,N);
    
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(2,N,QPSK_vec);
    n=(randn(N_rx,N)+1i*randn(N_rx,N))/sqrt(2);
    
    for kk=1:N,
        % Signal generation
        H=(randn(N_rx,2)+1i*randn(N_rx,2))/sqrt(2);
        
        [U,D,V]=svd(H); 
        
        H_tilde=H*V; % SVD Precoding 
        
        y=H_tilde*1/sqrt(2)*s(:,kk)+rho*n(:,kk);
        
        % Signal Detection ZF
        s_hat_ZF(:,kk)=pinv(1/sqrt(2)*H_tilde)*y;
        
        % Signal Detection ML
        
        DiffVec=repmat(y,1,16)-1/sqrt(2)*H_tilde*CombinationMat;
        CostVec=sum(abs(DiffVec).^2);
        
        [MinValue,MinLoc]=min(CostVec);
        s_tilde_ML(:,kk)=CombinationMat(:,MinLoc);
    end;
    
    s_tilde_ZF=sign(real(s_hat_ZF))/sqrt(2)+1i*sign(imag(s_hat_ZF))/sqrt(2);
    
    % Count error on each stream independently 
    NumOfErrorsZF1=sum(abs(s_tilde_ZF(1,:)-s(1,:))>eps);
    NumOfErrorsZF2=sum(abs(s_tilde_ZF(2,:)-s(2,:))>eps);
    SER_ZF(:,k)=[NumOfErrorsZF1; NumOfErrorsZF2]/N;
    
    NumOfErrorsML1=sum(abs(s_tilde_ML(1,:)-s(1,:))>eps);
    NumOfErrorsML2=sum(abs(s_tilde_ML(2,:)-s(2,:))>eps);
    SER_ML(:,k)=[NumOfErrorsML1;NumOfErrorsML2] /N;
    
end;

semilogy(SNRdB,SER_ZF(1,:)); grid;
hold on
semilogy(SNRdB,SER_ZF(2,:),'g'); 

semilogy(SNRdB,SER_ML(1,:),'r+');
semilogy(SNRdB,SER_ML(2,:),'k+');
xlabel('SNR(dB)');
ylabel('SER');
title(['Precoded SM ',num2str(N_rx),'X2 Rayleigh']);
legend('ZF stream 1','ZF stream 2','ML stream 1','ML stream 2')