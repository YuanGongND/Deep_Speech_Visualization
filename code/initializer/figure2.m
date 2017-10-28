function [ ] = figure1(  )

font_size = 20;
time = [ 0: 6/96000: 6 ];
time = time( 1: 96000 );
Fs = 16000;

%% real signal 
[ signal, Fs ] = audioread( 'audio/00003.wav' );

hold off;
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'originalSignal.png' );

hold off;
font_size_spec = 15;
showSpec( signal', Fs );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'originalSignalSpec.png' );

%% filter1 
font_size = 10;
N = 256; % the length of each filter

type = 'low'
filter = fir1( N -1, 0.5, type );

hold off;
plot( filter )
set(gcf, 'Position', [0, 0, 250, 250]);
xlim( [ 1, N ] );
xlabel('Point Index','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter1.png' );

hold off;
fqz = [ -Fs/2: Fs/N: Fs/2 ];
fqz = fqz( 1: N );
plot( fqz, fftshift( abs( fft( filter ) ) ) );
set(gcf, 'Position', [0, 0, 250, 250]);
xlabel('Frequency','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter1Spec.png' );

filter1 = filter;

%% filter2
type = 'high'
filter = fir1( N -1, 0.5, type );

hold off;
plot( filter )
set(gcf, 'Position', [0, 0, 250, 250]);
xlim( [ 1, N ] );
xlabel('Point Index','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter2.png' );

hold off;
fqz = [ -Fs/2: Fs/N: Fs/2 ];
plot( fqz, fftshift( abs( fft( filter ) ) ) );
set(gcf, 'Position', [0, 0, 250, 250]);
xlabel('Frequency','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter2Spec.png' );

filter2 = filter;


%% filter3
type = 'bandpass'
filter = fir1( N -1, [ 0.25, 0.75 ], type );

hold off;
plot( filter )
set(gcf, 'Position', [0, 0, 250, 250]);
xlim( [ 1, N ] );
xlabel('Point Index','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter3Pool.png' );

hold off;
fqz = [ -Fs/4: Fs/(2*N): Fs/4 ];
fqz = fqz( 1: N );
plot( fqz, fftshift( abs( fft( filter ) ) ) );
set(gcf, 'Position', [0, 0, 250, 250]);
xlabel('Frequency','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter3SpecPool.png' );

filter3 = filter;

%% filter4
type = 'high'
filter = fir1( N -1, 0.75, type );

hold off;
plot( filter )
set(gcf, 'Position', [0, 0, 250, 250]);
xlim( [ 1, N ] );
xlabel('Point Index','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter4Pool.png' );

hold off;
fqz = [ -Fs/4: Fs/(2*N): Fs/4 ];
plot( fqz, fftshift( abs( fft( filter ) ) ) );
set(gcf, 'Position', [0, 0, 250, 250]);
xlabel('Frequency','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
set(gca,'FontSize',font_size);
saveas( gcf, 'filter4SpecPool.png' );

filter4 = filter;

convOut1 = convFilterSignal( signal', filter1, 'convOut1' )
convOut2 = convFilterSignal( signal', filter2, 'convOut2' )
convOut3 = convFilterSignal( convOut1, filter3, 'convOut3' )
convOut4 = convFilterSignal( convOut2, filter4, 'convOut4' )

convOut6_1 = convFilterSignal( convOut1, filter2, 'convOut6_1' )
convOut6_2 = convFilterSignal( convOut2, filter1, 'convOut6_2' )

font_size = 20;
font_size_spec = 15;

convOut5 = convOut3 + convOut4;
signal = convOut5;
hold off;
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut5.png' );

hold off;
showSpec( signal, Fs );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut5Spec.png' );

convOut5 = convOut6_1 + convOut6_2;
signal = convOut5;
hold off;
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut6.png' );

hold off;
showSpec( signal, Fs );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut6Spec.png' );

%% pooling 1
convOut1Pool = maxpooling( convOut1 );
signal = convOut1Pool;
hold off;
time = [ 0: 6/48000: 6 ];
time = time( 1: 48000 );
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut1pool.png' );

hold off;
showSpec( signal, Fs/2 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut1poolSpec.png' );

%% pooling2
convOut2Pool = maxpooling( convOut2 );
signal = convOut2Pool;
hold off;
time = [ 0: 6/48000: 6 ];
time = time( 1: 48000 );
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut2pool.png' );

hold off;
showSpec( signal, Fs/2 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut2poolSpec.png' );

%% second layer
convOut3 = convFilterSignal( convOut1Pool, filter3, 'poolConvOut3' );
convOut4 = convFilterSignal( convOut2Pool, filter4, 'poolConvOut4' );

%% pooling 3
convOut3Pool = maxpooling( convOut3 );
signal = convOut3Pool;
hold off;
time = [ 0: 6/24000: 6 ];
time = time( 1: 24000 );
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut3pool.png' );

hold off;
showSpec( signal, Fs/4 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut3poolSpec.png' );

%% pooling 4
convOut4Pool = maxpooling( convOut4 );
signal = convOut4Pool;
hold off;
time = [ 0: 6/24000: 6 ];
time = time( 1: 24000 );
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'convOut4pool.png' );

hold off;
showSpec( signal, Fs/4 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'convOut4poolSpec.png' );

%% out 5
convOut5Pool = convOut3Pool + convOut4Pool;
signal = convOut5Pool;
hold off;
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'poolConvOut5.png' );

hold off;
showSpec( signal, Fs/4 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'poolConvOut5Spec.png' );

%% out6 
convOut6_1Pool = convFilterSignal( convOut1Pool, filter2, 'poolconvOut6_1' );
convOut6_2Pool = convFilterSignal( convOut2Pool, filter1, 'poolconvOut6_2' );

poolConvOut5 = maxpooling( convOut6_1Pool ) + maxpooling( convOut6_2Pool );
signal = poolConvOut5;
hold off;
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, 'poolConvOut6.png' );

hold off;
showSpec( signal, Fs/4 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, 'poolConvOut6Spec.png' );

end
%%
function [ output ] = maxpooling( input )
    output = [];
    i = 1;
    while i *2 <= size( input, 2 )
        output( i ) = max( [input( 2*i-1 ), input( 2*i )] );
        i = i + 1;
    end
end


%%
function [ signal ] = convFilterSignal( signal, filter, name )

convOut = [ ];
for len = 1: ( size( signal, 2 ) -  size( filter, 2 ) )
    convOut( len ) = sum( filter .* signal( len: len + size( filter, 2 ) - 1 ) );
end
Nout = size( convOut, 2 );

time = [ 0: 6/length( signal ): 6 ];
time = time( 1: length( signal ) );

font_size = 20;
hold off;
signal = [ zeros( 1, length(signal)-Nout ), convOut ];
plot( time, signal );
set(gcf, 'Position', [0, 0, 800, 250]);
xlabel('Time (s)','FontSize',font_size);
ylabel('Amplitude','FontSize',font_size);
ylim( [ -0.5 0.5 ] )
set(gca,'FontSize',font_size);
saveas( gcf, [name, '.png'] );

hold off;
font_size_spec = 15;
showSpec( convOut, 16000 );
set(gca,'FontSize',font_size_spec);
xlabel('Time (s)','FontSize',font_size_spec);
saveas( gcf, [name, 'spec.png'] );

end

%%
function [ ] = showSpec( recording, Fs )
    width = 256;
    height = 256;
    spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');
end