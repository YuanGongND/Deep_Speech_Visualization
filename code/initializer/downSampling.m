function [ ] = downSampling(  )

[original, Fs] = audioread( 'audio/00003.wav' );
%plot( original );
height = 1024;

% %% generate filter
% Ts = 1/16000;
% fFilter=3600;  %生成正弦波的频率（可修改）
% NFilter=1000; %采样点数
% t= -NFilter/2*Ts :Ts :NFilter/2*Ts-Ts;
% filter = sin( 2 *pi *fFilter *t); %最小正周期T=2π/|ω|   ω=2π*f
% plot( [0: NFilter - 1 ], filter );
% 
% fr=(-NFilter/2:NFilter/2  -1 )*Fs/NFilter;
% plot( fr, fftshift( abs( fft( filter ) ) ) );

%%

recording = original';
% recording = filter;

showSpec( recording, Fs, height );

for maxLayer = 1:5
    recording = maxpooling( recording );
    subplot( 1, 2, 1 )
    plot( recording );
    Fs = Fs /2;
    height = height /2;
    subplot( 1, 2, 2 );
    showSpec( recording, Fs, height );
end

end

function [ ] = showSpec( recording, Fs, height )
    width = 256;
    
    spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');
end

function [ output ] = maxpooling( input )
    output = [];
    i = 1;
    while i *2 <= size( input, 2 )
        output( i ) = mean( [input( 2*i-1 ), input( 2*i )] );
        i = i + 1;
    end
end