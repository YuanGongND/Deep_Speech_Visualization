function [ ] = getHandFeatures( recordingPath )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dirRecording = dir( recordingPath );
% the first two is not file, NOTICE, NOT SORTED BY NAME

setting.openSMILEPath = 'openSMILE'
%setting.featureFileName = [regexprep( recordingPath , '/|\.', ''),'.csv'];
setting.featureFileName = [regexprep( recordingPath , '|', ''),'.csv'];

% keep record of the mapping from feature of each segment to corresponding
% timeStamp

for fileIndex = 3 : length( dirRecording )
    recordingFile = dirRecording( fileIndex ); 
    % call openSmile extract the feature 
    % be cautious of space in the command 
    system( ...
       ['SMILExtract_Release -C ',...
        setting.openSMILEPath,...
        '/',...
        'IS09_emotion.conf -I ',...
        recordingPath,...
        '/',...
        recordingFile.name,...
        ' -O ',...
        setting.featureFileName ] );
    
end % end of processing all recording 

ConvertArffToCsv( setting );

end

