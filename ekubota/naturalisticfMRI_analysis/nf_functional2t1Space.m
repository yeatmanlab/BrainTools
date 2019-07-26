%function roi_timecourse = nf_functional2t1Space(subList)
subList = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184', 'PREK_1798','PREK_1302','PREK_1460','PREK_1110','PREK_1756',...
    'PREK_1966','PREK_1750','PREK_1940','PREK_1262','PREK_1113'};%,'PREK_1241'};

for ii = 1:length(subList)
    
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/t1');    
    functionalPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/func/GLMdenoise');
    
    cd(anatPath)
    % load in t1_acpc
    if exist(fullfile(anatPath,'t1_acpc.nii.gz'),'file')
        im3 = readFileNifti('t1_acpc.nii.gz');
    else 
        im3 = readFileNifti('t1_acpc_avg.nii.gz');
    end
    t1 = im3.data;
    t1_dimensions = im3.pixdim;
    % Load in alignment
    load tr.mat   
    
    %  get functional data 
    cd(functionalPath);
    scans = dir('denoisedGLMrun*.nii');
    nscans = size(scans,1);
    
    for ss = 1:nscans
        % use mean functional as reference 
        im1 = readFileNifti(fullfile(functionalPath,scans(ss).name));
        functional = im1.data;
        functional_dimensions = im1.pixdim(1:3);
        [outname ~] = strtok(scans(ss).name,'.');

        % Loop over timepoints
        for tt = 1:size(functional,4)
            func_t1(:,:,:,tt) = extractslices(t1, t1_dimensions, functional(:,:,:,tt), functional_dimensions, tr,1);
        end
        func_t1 = int16(func_t1);

        im3.data = func_t1;
        im3.fname = strcat('pre_',outname,'_t1.nii');
        writeFileNifti(im3);
    end 
end

return

corrmat = corr(roi_timecourse'); corrmat(corrmat == 1) = nan; nanmean(corrmat(:))