function [ recording, Fs ] = readAudioMono( fileName )

% read audio , transform it into single channel and change the shape
[ recording, Fs ] = audioread( fileName );
recording = ( recording( :, 1 ) + recording( :, 2 ) ) / 2;

% change the shape of the audio
recording = recording' ;

end

