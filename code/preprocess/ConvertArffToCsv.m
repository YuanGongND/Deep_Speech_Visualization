function [ output_args ] = ConvertArffToCsv( setting, featureNum )
% convert weka arff to csv
    if nargin < 2
        % then use all feature
        featureNum = -1; 
    end
    
    % arff file has a feature title, but matlab xlsread will only read
    % numerical information
    arffFile = xlsread( setting.featureFileName );
    %% arff numberical information = csv information
    csvFile = arffFile;
    csvwrite( setting.featureFileName, csvFile );
    
end

