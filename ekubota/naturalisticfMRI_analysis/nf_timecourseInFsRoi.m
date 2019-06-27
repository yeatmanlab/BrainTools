function timecourse_t1 = nf_timecourseInFsRoi(subList)


for ii = 1:length(subList)
    
    functionalPath = strcat('/mnt/scratch/PREK_Analysis/', subList{ii},'/ses-pre/func');
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/t1');    
    denoisedPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/func/GLMdenoise')
    roiPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/fsROIs')
    
    % Get betas from results.mat
    cd(functionalPath)
    
    % use meean functional as reference 
    im1 = readFileNifti(fullfile(denoisedPath,'denoisedGLMrun01.nii'));
    ref = mean(im1.data,4);
    inplane = mean(im1.data,4);
    inplane_dimensions = im1.pixdim(1:3);
    
    % get raw data to make nii too 
    cd(denoisedPath)
    im2 = readFileNifti(fullfile(denoisedPath,'denoisedGLMrun01.nii'))  
    timecourse = im2.data;
    
    cd(roiPath)
    im5 = readFileNifti('1007_ctx-lh-fusiform.nii.gz');
    lh_fus = im5.data;
    
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
    
    tmp = NaN(181,217,181,98);
    timecourse_t1 = NaN(length(subList),98);
    for tt = 1:size(timecourse,4)
        tmp(:,:,:,tt) = extractslices(lh_fus, t1_dimensions, timecourse(:,:,:,tt), inplane_dimensions, tr,1);
    end
    timecourse_t1(ii,:) = nanmean(nanmean(nanmean(tmp)));
end
