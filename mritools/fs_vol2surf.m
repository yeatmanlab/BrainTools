function fs_vol2surf(movingIm, subjid, hemi, outname)
% Map data in a nifti image to a subjects cortical surface

            % Map to Freesurfer surface
            cmd = sprintf('mri_vol2surf --mov %s --regheader %s --o %s --hemi %s --surf white --projfrac 1',...
                movingIm, subjid, outname, hemi);
            system(cmd)
