function nf_glmDenoiseSubs 

subList = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};
visitDates = {'20190525','20190525','20190525','20190525','20190525',...
    '20190524','20190524','20190525','20190525'};
for ii = 1:length(subList)
    dataDir = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/',...
        visitDates{ii},'/fmri/');
    nf_glmDenoise(dataDir)
end