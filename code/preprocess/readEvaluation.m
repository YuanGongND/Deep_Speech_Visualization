 function [ startTimeMatrix, endTimeMatrix, emotionMatrix, speakerMatrix, genderMatrix, fileNameMatrix ]...
    = readEvaluation( evalName )

%% read evaluation file

fidin = fopen( evalName, 'r' );
nline = 0;

startTimeMatrix = [ ];
endTimeMatrix = [ ];
emotionMatrix = [];
fileNameMatrix = [ ];
speakerMatrix = [ ];
genderMatrix = [ ];

while ~feof(fidin)        
   tline = fgetl(fidin);
   if size(tline,2)>0
     if tline(1)=='['
       %disp(tline);
       mark1=[];
       mark2=[];
       mark3=[];
       mark4=[];
       for i=1:size(tline,2)
         if tline(i)=='-'
             mark1=[mark1,i];
         end
         if tline(i)==']'
             mark2=[mark2,i];
         end
         if tline(i)=='['
             mark3=[mark3,i];
         end
         if tline(i)==','
             mark4=[mark4,i];
         end
       end
       
       start = tline(mark3(1)+1:mark1-1);
       ed = tline(mark1+1:mark2(1)-1);
       name = tline(mark2(1)+2:mark3(2)-6);
       clabel = tline(mark3(2)-4:mark3(2)-1);

       startTimeMatrix = [ startTimeMatrix; str2double( start ) ];
       endTimeMatrix = [ endTimeMatrix; str2double( ed ) ];
       fileNameMatrix = [ fileNameMatrix; name ];
       
       % encode emotion from text to number 0-3 (4 classes)
       categoricalEmotionCode = emotionEncode( clabel(1:3) );
       emotionMatrix = [ emotionMatrix ; categoricalEmotionCode ];
       
      % the speaker of this utterance 0-9 (10 classes)
      speakerIndex = speakerEncode( str2num( name( 5 ) ), name( end-3 ) );
      speakerMatrix = [ speakerMatrix; speakerIndex ]; 
      
      % the gender of this utterance 0-1 (2 classes)
      genderIndex = genderEncode( name( end - 3 ) );
      genderMatrix = [ genderMatrix; genderIndex ];
      
     end
   end
   nline = nline+1;
end

fclose(fidin);
