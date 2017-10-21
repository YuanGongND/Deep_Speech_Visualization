type = 'low';
N = 48;
b = fir1( N -1, 0.1, type );

plot( b )
plot( abs( fft( b ) ) );