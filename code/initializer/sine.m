clc;clear;

NSignal = 1024;
Fs = 16000;
Ts = 1 / Fs;
t = -NSignal/2*Ts:Ts:NSignal/2*Ts-Ts;
f = 1000;
signal = sinc( 2 *pi *t *f );

subplot( 3, 2, 1 )
plot( [ 0: NSignal - 1 ], signal );

fy = fft( signal );
fr=(-NSignal/2:NSignal/2  -1 )*Fs/NSignal;
%fy=fy(1:floor(end/2));
%fr=fr(1:floor(end/2));

subplot( 3, 2, 2 )
plot(fr,fftshift( abs(fy) ) );

%% generate filter
fFilter=600;  %生成正弦波的频率（可修改）
NFilter=64; %采样点数
t= -NFilter/2*Ts :Ts :NFilter/2*Ts-Ts;
filter = sin( 2 *pi *fFilter *t); %最小正周期T=2π/|ω|   ω=2π*f
hold on;
subplot( 3, 2, 3 )
plot( [0: NFilter - 1 ], filter );
subplot( 3, 2, 4 )
fr=(-NFilter/2:NFilter/2  -1 )*Fs/NFilter;
plot( fr, fftshift( abs( fft( filter ) ) ) );
%% conv
convOut = [];
for len = 1: (NSignal - NFilter)
    convOut( len ) = sum( filter .* signal( len: len + size( filter, 2 ) - 1 ) );
end
subplot( 3, 2, 5 )
Nout = size( convOut, 2 );
plot(  1: size( convOut, 2 ), convOut );
subplot( 3, 2, 6 )
fr=(-Nout/2:Nout/2  -1 )*Fs/Nout;
plot( fr, ifft( abs( fft( convOut ) ) ) );
