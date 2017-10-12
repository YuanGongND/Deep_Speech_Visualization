function [ genderIndex ] =genderEncode( gender )
% male = 0, female = 1

if strcmp( gender, 'M' ) == 1
   genderIndex = 0;
elseif strcmp( gender, 'F' ) == 1
   genderIndex = 1;
end

end

