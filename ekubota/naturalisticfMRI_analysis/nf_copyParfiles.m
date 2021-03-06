function nf_copyParfiles(subList,session)

% subList = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
%     'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};

parDir = '/mnt/disks/scratch/PREK_Analysis/code/BrainTools/ekubota/naturalisticfMRI_analysis';
for ii = 1:length(subList)
    dataDir = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subList{ii},'/',session,'/func/',...
        'Stimuli/parfiles');
    copyfile(fullfile(parDir,'run3.par'),dataDir)
    cd(dataDir)
    movefile('run3.par','run1.par');
    copyfile(fullfile(parDir,'run3.par'),dataDir)
    movefile('run3.par','run2.par');

end     
