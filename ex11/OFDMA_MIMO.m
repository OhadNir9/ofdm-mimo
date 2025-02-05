function OFDMA_MIMO
close all; clear; clc;


N_fft=256;
N=100; % Num of OFDM Symbols
CP_length=16;
NumRx=4;

STA0loc=[1:50]; % The SC indices of STA0
STA1loc=[51:70]; % The SC indices of STA1

% Freq Domain Data
s0=zeros(N_fft,N);
s0(STA0loc,:)=randsrc(length(STA0loc),N,[1+1i 1-1i -1+1i -1-1i])/sqrt(2); % STA0 Data (1st symbol is pilot)

s1_0=zeros(N_fft,N);
s1_0(STA1loc,:)=randsrc(length(STA1loc),N,[1+1i 1-1i -1+1i -1-1i])/sqrt(2); % STA1 Stream0 Data (1st symbol in pilot)
s1_0(:,2)=zeros;  % 2nd symbol is null
s1_1=zeros(N_fft,N);
s1_1(STA1loc,:)=randsrc(length(STA1loc),N,[1+1i 1-1i -1+1i -1-1i])/sqrt(2); % STA1 Stream1 Data (2nd symbol is pilot)
s1_1(:,1)=zeros;  % 1st symbol is null

%OFDM Modulator
% STA0
TimeDomainMat0=ifft(s0); % ifft is column-wise
TimeDomainMat_withCP0=[TimeDomainMat0(end-CP_length+1:end,:); TimeDomainMat0];
TimeDomainSingalLong0=TimeDomainMat_withCP0(:);

% STA1 (we avoid loops to make it explicit and clear. In a practical implementation do it with loops)
TimeDomainMat1_0=ifft(s1_0);
TimeDomainMat_withCP1_0=[TimeDomainMat1_0(end-CP_length+1:end,:); TimeDomainMat1_0];
TimeDomainSingalLong1_0=TimeDomainMat_withCP1_0(:);

TimeDomainMat1_1=ifft(s1_1);
TimeDomainMat_withCP1_1=[TimeDomainMat1_1(end-CP_length+1:end,:); TimeDomainMat1_1];
TimeDomainSingalLong1_1=TimeDomainMat_withCP1_1(:);

% Pass Through MIMO Channel
RxSignalLong=PassThroughChannel(TimeDomainSingalLong0.',NumRx,10,[0:CP_length-1]); % Pass STA0 through its channel 
RxSignalLong=RxSignalLong+PassThroughChannel([TimeDomainSingalLong1_0,TimeDomainSingalLong1_1].',NumRx,10,[0:CP_length-1]); % Add STA1 passed through its channel

% OFDMA Receiver
BigPostFFTRx=zeros(N_fft,N,NumRx); % Holds the FFT output for all Rx antennas (matrix per Rx antenna)

for k=1:NumRx,
    RxSignalMat=reshape(RxSignalLong(:,k),N_fft+CP_length,N);
    RxSignalMatWithoutCP=RxSignalMat(CP_length+1:end,:);
    PostFFTRx=fft(RxSignalMatWithoutCP); % fft is column-wise
    
    BigPostFFTRx(:,:,k)=PostFFTRx;
end;

% Channel Estimation
% STA0
ChannelEstimate0=zeros(NumRx,length(STA0loc));
for k=1:length(STA0loc),
    %temp=squeeze(BigPostFFTRx(STA0loc(k),1,:))./s0(STA0loc(k));
    ChannelEstimate0(:,k)=BigPostFFTRx(STA0loc(k),1,:)./s0(STA0loc(k),1)
end

% Plot one of the channels 
stem(abs(ChannelEstimate0(1,:)));
hold on 
stem(abs(ChannelEstimate0(2,:)),'r');
title('Estimated Channels of STA0'); 
xlabel('SC Num')
ylabel('abs(Channel)'); 
legend('Channel to Rx0','Channel to Rx1')

figure; 

% STA1
ChannelEstimate1_0=zeros(NumRx,length(STA1loc)); % For Stream0
ChannelEstimate1_1=zeros(NumRx,length(STA1loc)); % For Stream1

for k=1:length(STA1loc),
    ChannelEstimate1_0(:,k)=BigPostFFTRx(STA1loc(k),1,:)./s1_0(STA1loc(k),1); % Use 1st OFDM sym
    ChannelEstimate1_1(:,k)=BigPostFFTRx(STA1loc(k),2,:)./s1_1(STA1loc(k),2); % Use 2nd OFDM sym
end

% Demodulate All Payload
% STA0
Demodulated0=zeros(length(STA0loc),N);
for k=1:length(STA0loc),
    for kk=2:N, % First OFDM sym is pilot
        Demodulated0(k,kk)=ChannelEstimate0(:,k)'*squeeze(BigPostFFTRx(STA0loc(k),kk,:))/norm(ChannelEstimate0(:,k))^2; % MRC Processing
    end;
end;

Payload0=Demodulated0(:,2:end); plot(Payload0(:),'.');

% STA1
Demodulated1=zeros(length(STA1loc),N,2);
for k=1:length(STA1loc),
    CurrentMIMOChannel=[ChannelEstimate1_0(:,k),ChannelEstimate1_1(:,k)];
    
    for kk=3:N,% First 2 OFDM syms are pilots
        CurrentY=squeeze(BigPostFFTRx(STA1loc(k),kk,:));
        Demodulated1(k,kk,:)=pinv(CurrentMIMOChannel)*CurrentY;
    end;
end;

Payload1=Demodulated1(:,3:end,:);
Payload1_0=squeeze(Payload1(:,:,1)); 
Payload1_1=squeeze(Payload1(:,:,2));
hold on
plot(.99*Payload1_0(:),'r.'); % The .99 is just that we see all on the same plot...
plot(.98*Payload1_1(:),'k.'); 
title('Demodulated QAMs (Noiseless Version)');
legend('STA0','STA1 Stream0','STA1 Stream1')

%---------------------------------------------------------------------
function RxSignal=PassThroughChannel(TxSignal,NumRx,NumPaths,DelayRange)

NumTx=size(TxSignal,1);
PathsDelay=randsrc(NumPaths,1,DelayRange);
PathsPhase=exp(1j*2*pi*rand(NumPaths,1));

DoAs=2*pi*rand(NumPaths,1);
DoDs=2*pi*rand(NumPaths,1);

RxSignal=zeros(NumRx,size(TxSignal,2));

for k=1:NumPaths,
    for kk=1:NumTx,
        CurrentChunk=TxSignal(kk,:);
        RxSignal=RxSignal+ exp(-1i*pi*sin(DoDs(k))*(kk-1)) * PathsPhase(k)*exp(-1i*pi*sin(DoAs(k))*[0:NumRx-1].')*[zeros(1,PathsDelay(k)),CurrentChunk(1:end-PathsDelay(k))];
    end;
end;

RxSignal=RxSignal.';

