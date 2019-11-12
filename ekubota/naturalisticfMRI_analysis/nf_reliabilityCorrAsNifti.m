function nf_reliabilityCorrAsNifti(sublist)

for ii = 1:length(sublist)
    inplanePath = strcat('/mnt/scratch/PREK_Analysis/',sublist{ii},'/ses-pre/func');
    anatPath = strcat('/mnt/scratch/PREK_Analysis/',sublist{ii},'/ses-pre/t1');    
    
    reliability = nf_reliabilityCorrelation(sublist{ii});
    
    cd(inplanePath)

    % Load in inplane 
    im1 = readFileNifti('run01.nii');
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
    
   
    reliabilityCorr_t1= extractslices(t1, t1_dimensions, reliability, inplane_dimensions, tr,1);
    
 
    % write out contrast map as nifti
    im = readFileNifti(fullfile(anatPath,'t1_acpc.nii.gz'));
    
    im.data = reliabilityCorr_t1;
    im.descrip = 'reliabilityCorr';
    im.fname = fullfile(anatPath,'reliabilityCorrelation.nii.gz');
    writeFileNifti(im)
    
    load msh.mat
    
    % view on mesh with threshold.
    msh2 = AFQ_meshColor(msh, 'overlay',im, 'crange',[-.8 .8]);%,'thresh', .2);
    figure; ax = gca;
    colormap(jet)
    caxis([-.8 .8])
    [p,~,L]=AFQ_RenderCorticalSurface(msh2);

    view(180,-90); % Camera angle, ventral surface.
   % view(-90,0);
    camlight(L,'left'); % move the light. you can also give azymuth and elevation
    %axis off
    colorbar(ax)
    saveas(gcf,(fullfile(strcat('/home/ekubota/Desktop/nf_reliability/',sublist{ii},'_pre.png'))))
end 