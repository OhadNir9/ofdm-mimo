close all; clear; clc


QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);
SNRdB=0:2:30;
SNR=10.^(SNRdB./10);


SER_ML=zeros(size(SNRdB));
N=50000; % Number of pairs per SNR point

% Create a Matrix of 16 combinations for ML
CombinationLoc=zeros(2,16); 
CombinationLoc(1,:)=fix([0:15]/4)+1; 
CombinationLoc(2,:)=mod([0:15],4)+1; 

CombinationMat=QPSK_vec(CombinationLoc); 


for k=1:length(SNRdB),
    s_tilde_ML=zeros(2,N); 
    
    rho=1./sqrt(SNR(k));
    % Signal Generation
    s=randsrc(2,N,QPSK_vec);
    n=(randn(2,N)+1i*randn(2,N))/sqrt(2);
         
    for kk=1:N,
        % Signal generation
        H=(randn(2,2)+1i*randn(2,2))/sqrt(2);       
        
        y=H*1/sqrt(2)*s(:,kk)+rho*n(:,kk); 
               
        % Signal Detection ML
        DiffVec=repmat(y,1,16)-1/sqrt(2)*H*CombinationMat; 
        CostVec=sum(abs(DiffVec).^2); 
        
        [MinValue,MinLoc]=min(CostVec); 
        s_tilde_ML(:,kk)=CombinationMat(:,MinLoc); 
    end; 
      
    NumOfErrorsML=sum(abs(s_tilde_ML(:)-s(:))>eps); 
    SER_ML(k)=NumOfErrorsML/N/2; 

end;

semilogy(SNRdB,SER_ML,'r'); 
xlabel('SNR(dB)'); 
ylabel('SER'); 
title(['SM 2X2 Rayleigh']); 
legend('ML')