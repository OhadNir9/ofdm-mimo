close all; clear; clc;


N_fft=256;
N=100; % Num of OFDM Symbols
CP_length=16; 

IsGinie=0;  % 1 - use ideal channel estimate

SNRdB=40; 

% Freq Domain Data
s=randsrc(N_fft,N,[1+1i 1-1i -1+1i -1-1i])/sqrt(2); 

%OFDM Modulator
TimeDomainMat=ifft(s); % ifft is column-wise
TimeDomainMat_withCP=[TimeDomainMat(end-CP_length+1:end,:); TimeDomainMat]; 

TimeDomainSingalLong=TimeDomainMat_withCP(:); 

% Pass Through Channel 
h=zeros(16,1); h(1)=1; h(10)=0.5*1i; 

RxSingnalLong=conv(TimeDomainSingalLong,h); 
RxSingnalLong=RxSingnalLong(1:length(TimeDomainSingalLong)); 

% Add noise 
SignalPower=mean(abs(RxSingnalLong).^2);
RequiredNoisePower=SignalPower/10^(SNRdB/10);

NoiseVector=(randn(size(RxSingnalLong))+1i*randn(size(RxSingnalLong)))/sqrt(2)*sqrt(RequiredNoisePower); 
RxSingnalLongWithNoise=RxSingnalLong+NoiseVector; 

% OFDM Receiver
RxSignalMat=reshape(RxSingnalLongWithNoise,N_fft+CP_length,N);
RxSignalMatWithoutCP=RxSignalMat(CP_length+1:end,:); 

PostFFTRx=fft(RxSignalMatWithoutCP); % fft is column-wise 

% Channel Estimation (assuming we know the data of first OFDM symbol)
ChannelEstimation=PostFFTRx(:,1)./s(:,1); % Simple... Without any smoothing...

if IsGinie==1
    ChannelEstimation=fft(h,N_fft); % Use ideal channel estimation
end; 

% Demodulate Payload
DemodulatedPayload=PostFFTRx(:,2:end)./repmat(ChannelEstimation,1,N-1); 

plot(DemodulatedPayload(:),'.')

%EVM=10*log10(mean(abs(DemodulatedPayload(:)-(sign(real(DemodulatedPayload(:)))/sqrt(2)+1i*sign(imag(DemodulatedPayload(:)))/sqrt(2))).^2))
PayloadData=s(:,2:end); 
EVM=10*log10(mean(abs(DemodulatedPayload(:)-PayloadData(:) ).^2))

title(['SNR=',num2str(SNRdB),'dB. EVM=',num2str(EVM),'dB'])


