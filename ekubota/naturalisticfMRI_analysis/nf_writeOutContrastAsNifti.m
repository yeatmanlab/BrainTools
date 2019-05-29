
subList = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};
visitDates = {'20190525','20190525','20190525','20190525','20190525',...
    '20190524','20190524','20190525','20190525'};

for ii = 1:length(subList)
    
    betaPath = strcat('/mnt/scratch/PREK_Analysis/', subList{ii},'/',visitDates{ii},'/fmri');
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/nf_anatomy');    
    
    % Get betas from results.mat
    cd(betaPath)
    load results.mat
    betas = results.modelmd{2};
    npredictors = size(betas,4);
    
    % make a contrast map
    textvnontext = betas(:,:,:,2) - betas(:,:,:,3);
    text = betas(:,:,:,2);
    all = mean(betas(:,:,:,2:3),4)-betas(:,:,:,1);

    % use meean functional as reference 
    im1 = readFileNifti(fullfile(betaPath,'run01.nii'));
    ref = mean(im1.data,4);
    inplane = mean(im1.data,4);
    inplane_dimensions = im1.pixdim(1:3);
    
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
end 
    
%     load msh.mat
%     % view on mesh with threshold.
%     msh2 = AFQ_meshColor(msh, 'overlay',im, 'crange',[0 1],'thresh', .2);

