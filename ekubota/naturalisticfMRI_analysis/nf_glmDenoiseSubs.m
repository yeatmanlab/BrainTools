function nf_glmDenoiseSubs(subList,session)

% subList = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
%     'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};

for ii = 1:length(subList)
    dataDir = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subList{ii},'/',session,'/func');
    if exist(fullfile(dataDir,'run01.nii'),'file') && exist(fullfile(dataDir,'run02.nii'),'file')
        nf_glmDenoise(dataDir)
    end 
end