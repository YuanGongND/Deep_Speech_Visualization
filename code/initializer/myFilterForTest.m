function [ ] = myFilterForTest(  )
%%
for range = 0.1: 0.3

type = 'high';
N = 64;
filter = fir1( N -1, 0.3, type );

subplot( 3, 2, 1 )
plot( filter )

subplot( 3, 2, 2 )
plot( abs( fft( filter ) ) );

end

%% random initializer
% sig = sqrt( 2 / N );
% filter = normrnd( 0, sig,[ 1, N ]);
% 
% subplot( 3, 2, 1 )
% plot( filter )
% 
% subplot( 3, 2, 2 )
% plot( abs( fft( filter ) ) );

%%
% f1=250;f2=750;%待滤波正弦信号频率
% fs=2000;%采样频率
% t=0:(1/fs):0.1;
% signal=sin(2*pi*f1*t)+sin(2*pi*f2*t);%滤波前信号
% 
% subplot( 2,3,3 )
% plot( t,signal);%滤波前的信号图像

%% real signal 
[ signal, Fs ] = audioread( 'audio/00003.wav' );

convFilterSignal( signal', filter );
subplot( 3,2,3 )
plot( signal );

subplot( 3,2,5 )
showSpec( signal', Fs );

end

%%
function [ convOut ] = convFilterSignal( signal, filter )

convOut = [ ];
for len = 1: ( size( signal, 2 ) -  size( filter, 2 ) )
    convOut( len ) = sum( filter .* signal( len: len + size( filter, 2 ) - 1 ) );
end
Nout = size( convOut, 2 );
subplot( 3,2,4 )
plot(  1: size( convOut, 2 ), convOut );

subplot( 3,2,6 )
showSpec( convOut, 16000 )
% subplot( 3, 2, 6 )
% fr=(-Nout/2:Nout/2  -1 )*Fs/Nout;
% plot( fr, ifft( abs( fft( convOut ) ) ) );

end

%%
function [ ] = showSpec( recording, Fs )
    width = 256;
    height = 256;
    spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');
end