clc;clear;

filterIndex = 1;

for filterIndex = 1: 16
    for i = 1:1:26
        tempFile = csvread( [ 'test3/', num2str( i ), '_conv1', '.csv' ] );
        tempFilter = tempFile( filterIndex, : );
        tempFilter = real( fft( tempFilter ) );
        plot( tempFilter );
        hold on;
    end

    hold off;
end