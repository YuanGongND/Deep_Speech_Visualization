function [ ] = myFilterForTest(  )
%% real signal 
[ signal, Fs ] = audioread( 'audio/00003.wav' );

subplot( 3,2,3 )
plot( signal );

subplot( 3,2,5 )
showSpec( signal', Fs );

%%
bins = 64; % number of filters
N = 256; % the length of each filter
outputFilter = zeros( [ bins, N ] );
i = 1; % the filter index
for range = 0.0001: 0.95/bins: 0.95

type = 'bandPass';
filter = fir1( N -1, [ range, range + 0.95/bins ], type );

% type = 'low'
% filter = fir1( N -1, range, type );

subplot( 3, 2, 1 )
plot( filter )

subplot( 3, 2, 2 )
plot( abs( fft( filter ) ) );

% save for use in tensorflow
outputFilter( i, : ) = filter;
i = i + 1;

%convFilterSignal( signal', filter );
end
csvwrite( [ type, 'Filters_', num2str( N ), '_', num2str( bins ), '.csv' ], outputFilter );

%% random initializer
% sig = sqrt( 2 / N );
% filter = normrnd( 0, sig,[ 1, N ]);
% 
% subplot( 3, 2, 1 )
% plot( filter )
% 
% subplot( 3, 2, 2 )
% plot( abs( fft( filter ) ) );

%% all pass filter 
filter = zeros( [ 1, N ] );
filter( 1 ) = 1;

subplot( 3, 2, 1 )
plot( filter )

subplot( 3, 2, 2 )
plot( abs( fft( filter ) ) );

%% sample signal
% f1=250;f2=750;%待滤波正弦信号频率
% fs=2000;%采样频率
% t=0:(1/fs):0.1;
% signal=sin(2*pi*f1*t)+sin(2*pi*f2*t);%滤波前信号
% 
% subplot( 2,3,3 )
% plot( t,signal);%滤波前的信号图像


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