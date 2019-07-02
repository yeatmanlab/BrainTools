function roi_timecourse = nf_timecourseInFsRoi(subList)


for ii = 1:length(subList)
    
    functionalPath = strcat('/mnt/scratch/PREK_Analysis/', subList{ii},'/ses-pre/func');
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/t1');    
    denoisedPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/func/GLMdenoise');
    roiPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/fsROIs');
    
    % Get betas from results.mat
    cd(functionalPath);
    
    % use meean functional as reference 
    im1 = readFileNifti(fullfile(denoisedPath,'denoisedGLMrun01.nii'));
    ref = mean(im1.data,4);
    inplane = mean(im1.data,4);
    inplane_dimensions = im1.pixdim(1:3);
    
    % get raw data to make nii too 
    cd(denoisedPath);
    im2 = readFileNifti(fullfile(denoisedPath,'denoisedGLMrun01.nii'));
    functional = im2.data;
    
    cd(roiPath)
    im5 = readFileNifti('lh_ph1.nii.gz');
    roi = im5.data;
    
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
    
    roi_func = extractslices(roi, t1_dimensions, mean(functional,4), inplane_dimensions, tr,0,'nearest');
    % Loop over timepoints
    for tt = 1:size(functional,4)
        % Pull out 1 volume
        vol_f = squeeze(functional(:,:,:,tt));
        % Take the mean within the ROI for this timepoint
        tmp_f(tt) = nanmean(vol_f(roi_func==1));
    end
    % PUt the subjects mean timecourse into this variable
    roi_timecourse(ii,:) = tmp_f;
end
