function [ spectroEnergyFlat ] = wav2SpectrogramRGB( recording, Fs, width, height )
% convert waveform to spectrogram
logSign = 0;

if nargin <= 2
    width = 150;
    height = 64;
end

spectro = spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');
%spectrogram( recording, floor( size( recording, 2 )/( width /2 + 1) ) , floor( size( recording, 2 )/ ( width + 1 ) ) - 1, 2*height , Fs, 'yaxis');


if logSign == 1
   spectroEnergy =  log( abs( spectro ).^2 + 0.0001 );
else
   spectroEnergy =  abs( spectro ).^2 ;
end

% rotate the spectrogram to correct position
spectroEnergy = flipdim( spectroEnergy, 1 );

%## do not conduct normalization on a single spectrogram
%spectroEnergy = spectroEnergy/ ( max( max ( spectroEnergy ) ) - min( min ( spectroEnergy ) ) ) ;

% cut the edge
spectroEnergy = spectroEnergy( 1 : height, 1: width ); 

image( spectroEnergy * 256 );

%% flatten to one dimension, easier for further processing
%NOTICE: RESHAPE WILL ROTATE THE ORIGNAL FIGURE
spectroEnergyFlat = reshape( spectroEnergy', [ 1, height *width ] );

end

