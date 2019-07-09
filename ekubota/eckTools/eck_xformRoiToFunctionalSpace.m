function roiNifti_func = eck_xformRoiToFunctionalSpace(niftiRoi, niftiFunctional, outname)

%% Take a nifti roi and transform it to the functional space. 
% Inputs: 
%         nifti roi: 'roi.nii.gz' (in t1 space)
%
%         functional image: 'run01.nii.gz'
% 
%         outname: 'roi_func.nii.gz' 
%
% Outputs:
%         nifti roi in functional space
%

%% Read in nifti image of roi 
im = readFileNifti(niftiRoi);
roi = im.data;

t1_vox = im.pixdim; % get the voxel size

%% Read in functional image

im2 = readFileNifti(niftiFunctional);
functional = mean(im2.data,4); % take the mean functional (across time)

func_vox = im2.pixdim; % get the voxel size 

%% Transform roi 

 roi_func = extractslices(roi, t1_vox, functional, ...
     func_vox, tr,0,'nearest');
 
 %% Write out the roi in the functional space as a nifti 
 
roiNifti_func = im2;
roiNifti_func.fname = outname;
roiNifti_func.data = roi_func;
writeFileNifti(roiNifti_func);
