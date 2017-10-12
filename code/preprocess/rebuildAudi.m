folder = '../ex11 genderConvFinal\fm2\1advExample'
fileList = dir( folder );

for fileIndex = 3: length( fileList )
    wavFileName = fileList( fileIndex ).name;
    tempFile = csvread( [ folder, '/', wavFileName ] );
    tempFile = tempFile - 0.5;
    audiowrite( [ folder, '/', wavFileName, '.wav' ], tempFile, 16000 );
end