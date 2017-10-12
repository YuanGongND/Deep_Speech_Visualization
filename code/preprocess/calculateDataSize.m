function [] = calculateDataSize(  )

% waveform
audioLength = 6; % 
sampleRate = 16000; %
channelNum = 1;
audioBitNum = log2( 256 );

% spectrogram
timeResolution = 256;
frequencyResolution = 256;
specBitNum = log2( 256 );

% handCraftFeature (openSmile)
featureNum = 384;
precision = 64; % 

%% for waveform 
waveformSize = channelNum * sampleRate * audioBitNum * audioLength;
waveformSizeInKByte = convertBitToKByte( waveformSize );

%% for spectrogram 
spectrogramSize = timeResolution * frequencyResolution *specBitNum;
spectrogramSizeInKByte = convertBitToKByte( spectrogramSize );

%% for handCraft features
handCraftSize = featureNum *precision; 
handCraftSizeInKByte = convertBitToKByte( handCraftSize );

end

function [ sizeInKByte ] = convertBitToKByte( sizeInBit )
    sizeInKByte = sizeInBit / ( 1024 * 8 );
end