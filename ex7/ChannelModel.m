close all; clear; clc;

V=30;
f_c=2.5e9;

f_max=f_c*V/3e8;

Fs=f_max*40; % Oversampling - to allow looking at close enough temporal separations

% Create the Shaping Filter
N=4*1024; % FIR Length

% Constuct the Jakes PSD
FreqAxis=[0:N/2-1,  -N/2:-1]/N*Fs;
Sf=real(1./f_max./1./sqrt(1-(FreqAxis/f_max).^2));
Sf(Sf==inf)=0;

% The shaping filter
Gf=Sf.^(.5);
gt=fftshift(ifft(Gf));
gt=gt./norm(gt); % Normalize for unit variance

plot(linspace(0,Fs-Fs/length(Gf), length(Gf)),Gf); grid on;  
title('The Freq Response G(f)'); 
xlabel('Freq (Hz)'); figure; 
 
stem(gt); grid on; 
title('The Time Domain Filter')

% Generate the Fading Processes
PathPowerdB=[0 -1 -9 -10 -15 -20];
PowerCoeff=10.^(PathPowerdB./20);
PathDelay=[0 310 710 1090 1730 2510]*1e-9;


N_samples=100000;
N_samples_withGuards=N_samples+N-1;

a=zeros(length(PathPowerdB),N_samples);


for k=1:length(PathPowerdB);
    WhiteNoiseVec=(randn(1,N_samples_withGuards)+1i*randn(1,N_samples_withGuards))/sqrt(2);
    AfterGt=conv(WhiteNoiseVec,gt);
    a(k,:)=AfterGt(N:N_samples+N-1)*PowerCoeff(k); % Also add the right power to each process
end;


CoherenceTime=1/5/f_max;

ShortPeriod=CoherenceTime/8; ShortPeriodInSamples=fix(ShortPeriod*Fs);
LongPeriod=CoherenceTime*10;  LongPeriodInSamples=fix(LongPeriod*Fs);

% Compute the Channel Responses (usually through FFT, but here through the definition)

RequiredSamples=[1, 1+ShortPeriodInSamples,1+LongPeriodInSamples ];

FreqRange=[-10:.01:10]*1e6;
FreqResponses=zeros(length(RequiredSamples),length(FreqRange));

for k=1:length(RequiredSamples),
    for kk=1:length(PathPowerdB),
        FreqResponses(k,:)=FreqResponses(k,:)+a(kk,RequiredSamples(k))*exp(-1i*2*pi*PathDelay(kk).*FreqRange);
    end;
end;

figure; 
plot(FreqRange, abs(FreqResponses).'); grid;
legend('t_0','t_0+\Delta_1, \Delta_1<<T_c','t_0+\Delta_2, \Delta_2>>T_c')
xlabel('Freq'); ylabel('|H(f)|')