function nf_writeOutContrastAsNifti(subList)


for ii = 1:length(subList)
    
    betaPath = strcat('/mnt/scratch/PREK_Analysis/', subList{ii},'/ses-pre/func');
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/t1');    
    denoisedPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/func/GLMdenoise')
    
    % Get betas from results.mat
    cd(betaPath)
    load results.mat
    betas = results.modelmd{2};
    npredictors = size(betas,4);
    r2 = results.R2; 
    
    % make a contrast map
    textvnontext = betas(:,:,:,3) - betas(:,:,:,2);
    text = betas(:,:,:,3);
    all = mean(betas(:,:,:,2:3),4)-betas(:,:,:,1);

    % use meean functional as reference 
    im1 = readFileNifti(fullfile(betaPath,'run01.nii'));
    ref = mean(im1.data,4);
    inplane = mean(im1.data,4);
    inplane_dimensions = im1.pixdim(1:3);
    
    % get raw data to make nii too 
    cd(denoisedPath)
    im1 = readFileNifti(fullfile(denoisedPath,'denoisedGLMrun01.nii'))
    
    
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
    
   
    textvnontext_t1= extractslices(t1, t1_dimensions, textvnontext, inplane_dimensions, tr,1);
    all_t1= extractslices(t1, t1_dimensions, all, inplane_dimensions, tr,1);
    r2_t1 = extractslices(t1,t1_dimensions,r2,inplane_dimensions,tr,1);
 
    % write out contrast map as nifti
    im = readFileNifti(fullfile(anatPath,'t1_acpc.nii.gz'));
    
    im.data = all_t1;
    im.descrip = 'contrastMap';
    im.fname = fullfile(anatPath,'AllvBaseline.nii.gz');
    writeFileNifti(im)
    
    
    im.data = textvnontext_t1;
    im.descrip = 'contrastMap';
    im.fname = fullfile(anatPath,'TextvNontext.nii.gz');
    writeFileNifti(im)
    
    im.data = r2_t1;
    im.descrip = 'varianceExplained';
    im.fname = fullfile(anatPath,'ModelFit.nii.gz');
    writeFileNifti(im)
end 
    
%     load msh.mat
%     % view on mesh with threshold.
%     msh2 = AFQ_meshColor(msh, 'overlay',im, 'crange',[0 1],'thresh', .2);

