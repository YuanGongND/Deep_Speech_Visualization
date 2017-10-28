function [  ] = calculateResult( )

testResult = [];
trainResult = [];

for folderIndex = 0:4 
    folderName = [ 'folder', num2str( folderIndex ) ];
    absFolderName = [ folderName, '/accuracy.csv' ];
    accuracyFile = csvread( absFolderName );
    onTest = accuracyFile( 1, : );
    onTrain = accuracyFile( 2, : );
    testResult( folderIndex + 1 ) = max( onTest );
    trainResult( folderIndex + 1 ) = max( onTrain );

end

disp( mean( testResult ) );
disp( mean( trainResult ) );

