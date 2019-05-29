function fs_vol2surf(movingIm, subjid, hemi, outname, projdist)
% Map data in a nifti image to a subjects cortical surface
% 
% fs_vol2surf(movingIm, subjid, hemi, outname, [projdist])
%
% Use defaults unless projecdist is defined
%
% example:
% fs_vol2surf('/mnt/scratch/freesurfer/PreK_EK/roi3.nii.gz','PreK_EK','rh',...
% '/mnt/scratch/freesurfer/PreK_EK/roi3.mgz', [-1 4 1])


% Map to Freesurfer surface
if ~exist('projdis','var') || isempty(projdist)
    
    cmd = sprintf('mri_vol2surf --mov %s --regheader %s --o %s --hemi %s --surf white',...
        movingIm, subjid, outname, hemi);
else
    cmd = sprintf('mri_vol2surf --mov %s --regheader %s --o %s --hemi %s --surf white --projdist-avg %d %d %d',...
        movingIm, subjid, outname, hemi, projdist);
end
system(cmd)
