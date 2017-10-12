%% compare adv and original audio 
originalAudio = csvread( '../../processedData/datasetMiniTestNorm.csv' );
advAudio = csvread( '../ex3/adv0.3.csv' );

%%
selectedAudioSampeIndex = 501;
originalSample = originalAudio( selectedAudioSampeIndex, 1: 96000 );
advSample = advAudio( selectedAudioSampeIndex, 1: 96000 );

subplot( 3,1,1 )
plot( originalSample );
subplot( 3,1,2 )
plot( advSample );

audiowrite( 'advSample.wav', advSample, 16000 );
audiowrite( 'oriSample.wav', originalSample, 16000 );

%%
diff = advSample - originalSample;

%plot( diff )
envelopediff = diff.* originalSample;
advSample2 = originalSample + envelopediff;
subplot( 3, 1, 3 )
plot( advSample2 )
audiowrite( 'advSample2.wav', advSample2, 16000 );
%%
width = 256;
height = 256;
spectrogram( diff, floor( size( diff,2)/( width /2)), floor(size(diff,2)/ width ), 2*height , Fs, 'yaxis');

 