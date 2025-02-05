function OFDM_PAPR

close all; clear; clc;


Constellation=256; % 4,16,64,256,1024


N_fft=2048; 
N=10000; % Num of OFDM Symbols

OS_factor=4; 


OS_fft=N_fft*OS_factor; 

OneDimLimit=sqrt(Constellation)-1; 
% QAM Modulator (no need to normalize for PAPR computation...)
FreqDomainData=randsrc(N_fft,N,[-OneDimLimit:2:OneDimLimit])+1i*randsrc(N_fft,N,[-OneDimLimit:2:OneDimLimit]); 

% Over Sampling 
FreqDomainDataOverSampled=zeros(OS_fft,N); 
FreqDomainDataOverSampled(1:N_fft/2,:)=FreqDomainData(1:N_fft/2,:); 
FreqDomainDataOverSampled(end-N_fft/2+1:end,:)=FreqDomainData(end-N_fft/2+1:end,:); 

%OFDM Modulator
TimeDomainMat=ifft(FreqDomainDataOverSampled); % ifft is column-wise

% Compute the PAPR
PAPRdB=zeros(1,N); 

for k=1:size(TimeDomainMat,2),
    PAPRdB(k)=10*log10(max(abs(TimeDomainMat(:,k))).^2/mean(abs(TimeDomainMat(:,k)).^2));
end; 

[CDF1,SNRdBvec1]=MyCDF(PAPRdB); 
semilogy(SNRdBvec1,CDF1); grid; 
xlabel('PRPR_0(dB)')
ylabel('Prob(PAPR>PAPR_0)'); 
title(['PAPR in an OFDM system with ',num2str(N_fft),' SCs and QAM',num2str(Constellation)]); 

%-------------------------------------------------

function [cdfout,SNRdBvec]= MyCDF(data)
SNRdBvec=1:.01:(max(data)+1);

for k=1:length(SNRdBvec);
    cdfout(k)=sum(data>SNRdBvec(k));
end;
cdfout=cdfout/length(data);
