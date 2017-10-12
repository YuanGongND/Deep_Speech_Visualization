clc;clear;

% some constant
precision = 'int8';
sampleRate = 16000; 
CUTLENGTH = 6; % 6 seconds
WAVSIZE = sampleRate *CUTLENGTH; % sample rate *record length
specHeigth = 256;
specWidth = 256;

% fix the random seed to prove reproducibility
rng(3);

%% for each session
for sessionIndex = 1:5
    
    % initialize the data set 
    waveformSet = zeros( 1500, WAVSIZE + 4 ); % +1 speaker label, +2 emotion label, +3 gender label, +4 speech length
    specSet = zeros( 1500, specHeigth *specWidth + 4 );
    dataIndex = 1;
    
    % get the dir of wav file and label file 
    wavDir = [ '../../data/IEMOCAP_full_release/session', num2str( sessionIndex ), '/dialog/wav/' ];
    labelDir = [ '../../data/IEMOCAP_full_release/session', num2str( sessionIndex ), '/dialog/EmoEvaluation/' ];
    
    % for each file
    wavFile=dir( wavDir );
    for fileIndex = 3: length( wavFile ) % 3 because the first two are dicts
        wavFileName = wavFile( fileIndex ).name
        
        % if it is a eligible wav file
        if isequal( wavFileName( 1:3 ), 'Ses' ) && isequal( wavFileName( end-3: end ), '.wav' )
           [ recording, Fs ] = readAudioMono( [ wavDir, wavFileName ] );
           testName = wavFileName( 1: end - 4 );
           labelName = [ testName, '.txt' ];
           [ startTimeMatrix, ...
             endTimeMatrix, ...
             categoricalLabelMatrix, ...
             speakerMatrix, ...
             genderMatrix, ...
             fileNameMatrix ] = readEvaluation( [ labelDir, labelName ] );
           
           % for each utterance 
           for utteranceIndex = 1: length( startTimeMatrix )
               audioClip =  processAudio( recording, Fs, startTimeMatrix( utteranceIndex ), endTimeMatrix( utteranceIndex ), CUTLENGTH ); % in [ -1, 1]
               audioSpec = wav2SpectrogramRGB( audioClip, Fs ); % in [ 0, 1 ]
               
               if exist( [ '../../processedData/rawWav/', num2str(sampleRate), '/', num2str( sessionIndex ) ] ) == 0
                  mkdir( [ '../../processedData/rawWav/', num2str(sampleRate), '/', num2str( sessionIndex ) ] );
               end
               % record the raw audio
               audiowrite( [ '../../processedData/rawWav/', num2str(sampleRate),'/', num2str( sessionIndex ), '/', num2str( dataIndex, '%05d' ), '.wav'], audioClip , Fs );
               
               % rescale waveform and spectrogram (after rewrite to short-cut audio for extracting handcraft feature, won't affect handcraft feature extraction)
               waveformSet( dataIndex, : ) = [ audioClip, categoricalLabelMatrix( utteranceIndex ), speakerMatrix( utteranceIndex ), genderMatrix( utteranceIndex ), ( endTimeMatrix( utteranceIndex ) - startTimeMatrix( utteranceIndex ) ) ];
               specSet( dataIndex, : ) = [ audioSpec, categoricalLabelMatrix( utteranceIndex ), speakerMatrix( utteranceIndex ), genderMatrix( utteranceIndex ), ( endTimeMatrix( utteranceIndex ) - startTimeMatrix( utteranceIndex ) ) ];
               dataIndex = dataIndex + 1;
           end
        end
    end
   
   % write the waveform file for each session ( original )
%    if exist( [ '../../processedData/waveform/', num2str(sampleRate) ] ) == 0
%        mkdir( [ '../../processedData/waveform/', num2str(sampleRate) ] );
%    end
%    if exist( [ '../../processedData/spectrogram/', num2str(sampleRate) ] ) == 0
%        mkdir( [ '../../processedData/spectrogram/', num2str(sampleRate) ] );
%    end
%    
%    csvwrite( [ '../../processedData/waveform/', num2str(sampleRate), '/session_', num2str( sessionIndex ), '.csv' ],waveformSet( 1:dataIndex-1, : ) );
%    csvwrite( [ '../../processedData/spectrogram/', num2str(sampleRate), '/session_', num2str( sessionIndex ), '.csv' ],specSet( 1:dataIndex-1, : ) );
   
   % write the waveform file for each session ( int8 )
   precision = 'int8';
   
   if exist( [ '../../processedData/waveform/', num2str(sampleRate), '_', precision ] ) == 0
       mkdir( [ '../../processedData/waveform/', num2str(sampleRate), '_', precision ] );
   end
   if exist( [ '../../processedData/spectrogram/', num2str(sampleRate), '_', precision ] ) == 0
       mkdir( [ '../../processedData/spectrogram/', num2str(sampleRate), '_', precision ] );
   end
   
   waveformSetInt8 = changePrecision( waveformSet, 'waveform', 'int8' );
   csvwrite( [ '../../processedData/waveform/', num2str(sampleRate), '_', precision, '/session_', num2str( sessionIndex ), '.csv' ],waveformSetInt8( 1:dataIndex-1, : ) );
   specSetInt8 = changePrecision( specSet, 'spectrogram', 'int8' );
   csvwrite( [ '../../processedData/spectrogram/', num2str(sampleRate), '_', precision, '/session_', num2str( sessionIndex ), '.csv' ],specSetInt8( 1:dataIndex-1, : ) );
    
end

% get handcraft features 
baseFolder = { [ '../../processedData/rawWav/',  num2str( sampleRate ) ] };
folder = {  '1', '2', '3', '4', '5'  };
batchGetHandFeature( baseFolder, folder );

% move handcraft file to handcraft folder
if exist( [ '../../processedData/handCraft/', num2str(sampleRate) ] ) == 0
   mkdir( [ '../../processedData/handCraft/', num2str(sampleRate) ] );
end
for fileIndex = 1: 5
    movefile( [ '../../processedData/rawWav/', num2str( sampleRate ), '/', folder{ fileIndex }, '.csv' ], ...
        [ '../../processedData/handCraft/', num2str(sampleRate) ,'/', folder{ fileIndex }, '.csv' ] );
end


