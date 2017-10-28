function [  ] = calculateResult( inputDir )

for folderIndex = 0:4 
    folderName = [ 'folder', num2str( folderIndex ) ];
    absFolderName = [ inputDir, '/', folderName, '/accuracy.csv' ];
    accuracyFile = csvread( absFolderName );
    onTest = accuracyFile( :, 1 );
    onTrain = accuracyFile( :, 2 );
    disp( max( onTest ) );
    disp( max( onTrain ) );
end

