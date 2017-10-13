function [ outputRecording, Fs ] = processAudio( wholeRecording, Fs, startTime, endTime, cutLength )

% cut/pad audio into required length
startPoint = ceil( startTime *Fs );
endPoint = floor( endTime *Fs );

recording = wholeRecording( startPoint: endPoint );

if size( recording, 2 ) > cutLength *Fs
    outputRecording = recording( 1: cutLength *Fs );
else 
    % pad with zeros at the end
    outputRecording = zeros( 1, cutLength *Fs );
    outputRecording( 1: size( recording, 2 ) ) = recording';
end

% normalize audio to make each audio between -1 and 1 (optional)
%## do not normalize single waveform    
%outputRecording = normalizeAudio( outputRecording );

end

