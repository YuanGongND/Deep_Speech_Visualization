clc;clear;
filterIndex = [ 9, 25, 46, 62 ];
font_size = 10;
fsize1 = 10

Fs = 16000;
N = 256;
fqz = [ -Fs/2: Fs/N: Fs/2 ];
fqz = fqz( 1: N );

%% the initial states
i = 1;
init = csvread( 'bandPassFilters_256_64.csv' );
for index = filterIndex
    tempfilter = init( index, : );
    subplot( 4, 4, i )
    plot( tempfilter )
    xlim([1,256])
    i = i + 1;
    
    xlabel('Point Index','FontSize',font_size);
    ylabel('Amplitude','FontSize',font_size);
    title( ['The Initial Filter ', num2str(i-1) ], 'fontSize',fsize1  );
end

%% the fft of initial
init = csvread( 'bandPassFilters_256_64.csv' );
for index = filterIndex
    tempfilter = init( index, : );
    subplot( 4, 4, i )
    plot( fqz, fftshift( abs( fft( tempfilter ) ) ) );
    xlim([-Fs/2, Fs/2])
    i = i + 1;
   
    xlabel('Frequency (Hz)','FontSize',font_size);
    ylabel('Amplitude','FontSize',font_size);
    title( ['Spectrum of the Initial Filter ', num2str(i-5) ], 'fontSize',fsize1  );
end

%% the last states
filterIndex = filterIndex - 1;
for index = filterIndex
    title1 = [ 'convFilter/conv1_100_', num2str( index ), '.csv' ];
    tempfilter = csvread( title1 );
    subplot( 4,4,i )
    plot( tempfilter )
    xlim( [1,256] )
    i = i + 1;
    
    xlabel('Point Index','FontSize',font_size);
    ylabel('Amplitude','FontSize',font_size);
    title( ['Trained Filter ', num2str(i-9) ], 'fontSize',fsize1  );
end

%% the last states
filterIndex = filterIndex ;
for index = filterIndex
    title1 = [ 'convFilter/conv1_100_', num2str( index ), '.csv' ];
    tempfilter = csvread( title1 );
    subplot( 4,4,i )
    plot( fqz, fftshift( abs( fft( tempfilter ) ) ) );
    xlim([-Fs/2, Fs/2])
    i = i + 1;
    
    xlabel('Frequency (Hz)','FontSize',font_size);
    ylabel('Amplitude','FontSize',font_size);
    title( ['Spectrum of the Trained Filter ', num2str(i-14) ], 'fontSize',fsize1 );
end

set(gcf, 'Position', [0, 0, 1000, 800]);
saveas(gcf,'showFilter','epsc');