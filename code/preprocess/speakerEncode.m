function [ speakerIndex ] = speakerEncode( session, gender )
% encode the speakers of IEMOCAP to number 0 - 9 ( 10 speakers in total ),
% start from male 

if strcmp( gender, 'M' ) == 1
   speakerIndex = ( session - 1 ) *2 + 0;
elseif strcmp( gender, 'F' ) == 1
   speakerIndex = ( session - 1 ) *2 + 1;
end

end

