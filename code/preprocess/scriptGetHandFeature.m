folders = {'noise6'};
subFolders = {'noise6_80','noise6_85','noise6_90','noise6_95','noise6_100','noise6_105'};
tt = [80,85,90,95,100,105];
%subsubFolders = {'rawwav5','rawwav4','rawwav3','rawwav2'};
subsubFolders = [1,2,3,4,5];

for f1 = 1:size(folders,2)
    for f2 = 1:size(subFolders,2)
        for f3 = 1:size(subsubFolders,2)
            processDir = [folders{f1},'/',subFolders{f2},'/','rawwav',num2str(tt(f2)+f3)]
            getHandFeatures( processDir );
        end
    end
end

% for f1 = 1:size(folders,2)
%     for f2 = 1:size(subFolders,2)
%         for f3 = 1:size(subsubFolders,2)
%             processDir = [folders{f1},'/',subFolders{f2},'/',subsubFolders{f3}];
%             getHandFeatures( processDir );
%         end
%     end
% end