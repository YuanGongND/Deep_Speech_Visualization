function [ outputMatrix ] = changePrecision( inputMatrix, dataType, precision )
% convert the matrix to desired precision

data = inputMatrix( :, 1: size( inputMatrix, 2 ) - 5 );
label = inputMatrix( :, size( inputMatrix, 2 ) - 5 + 1 : size( inputMatrix, 2 ) );

if strcmp( precision, 'int8' ) == 1
    if strcmp( dataType, 'waveform' ) == 1
        % orginally in [-1,1], rescale it to 0-255
         data = ( data + 1 ) /2 *255; % rescale to 0-255   
         data = uint8( data );
    end
    
    if strcmp( dataType, 'spectrogram' ) == 1
        % originally in [ 0, 1 ], rescale to 0 -255
        data = data * 255;
        data = uint8( data );
    end
end

outputMatrix = [ data, label ];

