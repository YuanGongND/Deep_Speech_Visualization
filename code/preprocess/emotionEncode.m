function [ code ] = emotionEncode( emotion )
% ang = 0, hap = 1, neu = 2, sad = 3

emotionCandidate = { 'ang', 'hap', 'neu', 'sad', 'exc' };

% explicitly show the result 
if strcmp( emotion, 'ang' ) == 1
    code = 0;
elseif strcmp( emotion, 'hap' ) == 1
    code = 1;
elseif strcmp( emotion, 'exc' ) == 1
    code = 1;
elseif strcmp( emotion, 'neu' ) == 1
    code = 2;
elseif strcmp( emotion, 'sad' ) == 1
    code = 3;
else 
    code = 4; % emotion is not interested in our research
end
   
end

