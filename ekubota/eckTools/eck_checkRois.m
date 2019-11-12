%D = getPreInterventionSubs;

anatdir = fullfile('/mnt/scratch/LMB_Analysis/',D.subject{ii},'mrVista_Anat');
cd(anatdir)

rois = {'lh_wvf_p3_ots.nii.gz','rh_wvf_p3_ots.nii.gz',... 
    'lh_wvf_p3_cos.nii.gz','rh_wvf_p3_cos.nii.gz'};

load(fullfile(anatdir,'msh.mat'));
map ='wordwordVadultchild_parametric.nii.gz';
fill_range = [3 4];

% visualize old roi on the mesh
im_glm = niftiRead(fullfile(anatdir,map));
msh3 = AFQ_meshColor(msh, 'overlay',im_glm, 'thresh',[3 4],'crange',[3 4],'cmap','autumn');
[~,~,L]=AFQ_RenderCorticalSurface(msh3)
view(180,-90);camlight(L,'left');
rgbvals = winter(4);

for rr = 1:length(rois)
    if exist(fullfile(anatdir,rois{rr}),'file')
        r_roi_old = niftiRead(fullfile(anatdir,rois{rr}));
        roiMesh(rr) = AFQ_meshCreate(r_roi_old,'color',rgbvals(rr,:),'smooth',1);
        patch(roiMesh(rr).tr,'FaceAlpha',.3,'EdgeAlpha',.3); shading('interp');
    end 
end
movegui('northwest')


% define roi on mesh in a new window, if it needs to be redefined.
[~,vRoi] = AFQ_meshDrawRoi(msh, [], fullfile(anatdir,map), fill_range,[], [],[],[0 1 2 3]);
    


