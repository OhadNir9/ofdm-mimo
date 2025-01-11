close all; clear; clc

N_rx=2;

QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);

if N_rx<3
    SNRdB=0:2:42;
else
    SNRdB=0:2:30;
end;
SNR=10.^(SNRdB./10);

SER_ZF=zeros(size(SNRdB));
SER_ML=zeros(size(SNRdB));
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
        
        y=H*1/sqrt(2)*s(:,kk)+rho*n(:,kk);
        
        % Signal Detection ZF
        s_hat_ZF(:,kk)=pinv(1/sqrt(2)*H)*y;
        
        % Signal Detection ML
        
        DiffVec=repmat(y,1,16)-1/sqrt(2)*H*CombinationMat;
        CostVec=sum(abs(DiffVec).^2);
        
        [MinValue,MinLoc]=min(CostVec);
        s_tilde_ML(:,kk)=CombinationMat(:,MinLoc);
    end;
    
    s_tilde_ZF=sign(real(s_hat_ZF))/sqrt(2)+1i*sign(imag(s_hat_ZF))/sqrt(2);
    
    NumOfErrorsZF=sum(abs(s_tilde_ZF(:)-s(:))>eps);
    SER_ZF(k)=NumOfErrorsZF/N/2;
    
    NumOfErrorsML=sum(abs(s_tilde_ML(:)-s(:))>eps);
    SER_ML(k)=NumOfErrorsML/N/2;
    
end;

semilogy(SNRdB,SER_ZF); grid;
hold on
semilogy(SNRdB,SER_ML,'r');
xlabel('SNR(dB)');
ylabel('SER');
title(['SM ',num2str(N_rx),'X2 Rayleigh']);
legend('ZF','ML')