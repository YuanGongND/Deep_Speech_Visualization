test1 = ones( 2000, 96000 ) *255;
test1 = uint8( test1 );
csvwrite( 'test.csv', test1 );
dlmwrite('test.csv', test1, 'delimiter', ',', 'precision', 0); 