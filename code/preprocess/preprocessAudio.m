originalData = csvread( '../ex13 emotionConvFinal\outputShuffleDataset\dataSetAfterShuffle.csv' );
trainFeature = originalData( 1: 1800, 1:96000 );
trainLabel = originalData( 1:1800, 96001 );

testFeature = originalData( 1801: 2000, 1: 96000 );
testLabel = originalData( 1801: 2000, 96001 );

trainFeature = trainFeature/2;
testFeature = testFeature/2;

%%
for i = 1: size( trainFeature, 1 )
    tempAudio = trainFeature( i, : );
    audiowrite( ['train/',num2str(i,'%05d'),'.wav'], tempAudio, 16000 );
end

for i = 1: size( testFeature, 1 )
    tempAudio = testFeature( i, : );
    audiowrite( ['test/',num2str(i,'%05d'),'.wav'], tempAudio, 16000 );
end

%%
if exist('train.csv','file') ~= 0
    delete( 'train.csv' );
end
 getHandFeatures( 'train' );

if exist( 'test.csv','file' ) ~= 0
    delete('test.csv'); 
end
getHandFeatures( 'test' );

%% add a name to feature, for weka selection
trainFile = csvread( 'train.csv' );
trainFile = [ trainFile, trainLabel ];
csvwrite('trainWithLabel.csv', trainFile);
trainFile = [ 1:size(trainFile,2); trainFile ];
csvwrite( 'trainWeka.csv', trainFile );

testFile = csvread( 'test.csv' );
testFile = [ testFile, testLabel ];
csvwrite( 'testWithLabel.csv', testFile );
testFile = [ 1:size(testFile,2); testFile ];
csvwrite( 'testWeka.csv', testFile );

%% generate all versions of testing data
folderName = { '_1adv.csv', '_1noise.csv' };
for folder = 1: 2
for eps = 0.031:0.001:0.039
   if eps == 0
       epsString = '0.0';
   else 
       epsString = num2str( eps );
   end
   
   if ~exist( [ epsString, folderName{folder}(3:5) ] ) 
       mkdir( [ epsString, folderName{folder}(3:5) ] );
   end
   tempTestFile = csvread( [ 'data/', epsString, folderName{ folder } ] ) - 0.5;
   
   for i = 1: size( tempTestFile, 1 )
       tempAudio = tempTestFile( i, : );
       audiowrite( [ epsString, folderName{folder}(3:5), '/',num2str( i,'%05d'),'.wav'], tempAudio, 16000 );
   end
   
   if exist( [epsString, folderName{folder}(3:5), '.csv'] ,'file') ~= 0
       delete( [epsString, folderName{folder}(3:5) , '.csv'] );
   end
   getHandFeatures( [ epsString, folderName{folder}(3:5) ] );
   tempFile = csvread( [epsString, folderName{folder}(3:5), '.csv'] );
   tempFile = [ tempFile, testLabel ];
   csvwrite( [ epsString, folderName{folder}(3:5), '.csv' ], tempFile );
end 
end


