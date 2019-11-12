% Change to subject directory
cd /mnt/scratch/PREK_Analysis/PREK_1208/ses-pre/t1/
% Load segmentation
t1 = niftiRead('t1_class.nii.gz');
% Make a mesh and smooth it to look like a brain
braincolor = [.9 .8 .7]
% msh = AFQ_meshCreate(t1,'boxfilter', 5, 'color', braincolor);

load msh.mat
% Load up each nifti roi as a vista roi
roiNames =     {'lh_itg1.nii.gz','lh_itg2.nii.gz','lh_itg3.nii.gz','lh_itg4.nii.gz','lh_fus1.nii.gz','lh_fus2.nii.gz','lh_fus3.nii.gz','lh_fus4.nii.gz'}; %{'1007_ctx-lh-fusiform.nii.gz','1009_ctx-lh-inferiortemporal.nii.gz'};

rgbvals = jet(8);% rgbvals = [0 .5 .7;.7 0 0; 0 .5 .7; .7 0 0]; %; .7 0 0];
msh2 = msh;
dilateRoi=2;
for ii = 1:length(roiNames)
    if exist(fullfile('/mnt/scratch/PREK_Analysis/PREK_1208/fsROIs',roiNames{ii}),'file')
        roi = dtiRoiFromNifti(fullfile('/mnt/scratch/PREK_Analysis/PREK_1208/fsROIs',roiNames{ii}),[],[],'mat')
        msh2 = AFQ_meshAddRoi(msh2, roi, rgbvals(ii,:), dilateRoi);
    end
end
figure;
[p,~,L]=AFQ_RenderCorticalSurface(msh2); % Render cortex
view(180,-90); % Camera angle, ventral surface.
%view(-90,0);
camlight(L,'left'); % move the light. you can also give azymuth and elevation
axis off % turn of plot axis

%print('distributedAndOverlapping.png','-dpng','-r300')
% %% To put a heatmap
% t1 = niftiRead('t1_class.nii.gz');
% msh = AFQ_meshCreate(t1,'boxfilter', 5);
% AFQ_RenderCorticalSurface(msh);
% im = niftiRead('functionalOverlay-08-Mar-2018.nii.gz');
% im.fname = 'Objects.nii.gz'
% im.data=smooth3(im.data,'gaussian',9);
% msh2 = AFQ_meshColor(msh, 'overlay',im, 'thresh',[3 5],'crange',[3 5],'cmap','autumn');
% [~,~,L]=AFQ_RenderCorticalSurface(msh2)
% view(180,-90)
% camlight(L,'left')