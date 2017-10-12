function [ outputData ] = normalizeAudio( inputData )
% Normalize each row (sample) to -1 and 1 (min and max)
    data = inputData;
    maxOfAudio = max( abs( data ),[], 2 );
    augFactor = 1./ maxOfAudio;
    data = data.* augFactor;
    outputData = data;
end

