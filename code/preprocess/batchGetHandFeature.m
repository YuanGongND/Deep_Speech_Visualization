function [ ] = batchGetHandFeatures( folders, subFolders )

if nargin == 0
    disp( 'no arg' )
    folders = {'noise6'};
    subFolders = {'noise6_80','noise6_85','noise6_90','noise6_95','noise6_100','noise6_105'};
end

for f1 = 1:size( folders, 2 )
    for f2 = 1: size( subFolders, 2 )
        processDir = [ folders{f1}, '/', subFolders{f2} ]
        getHandFeatures( processDir );
    end
end

end