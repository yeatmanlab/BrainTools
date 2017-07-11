
% Function designed to take an fsgd file and run a cross-sectional analysis
% on cortical thickness data using a given contrast.
% The function runs the mris_preproc, mri_surf2surf, and mri_glmfit
% functions for both hemispheres, producing files which can be viewed in
% freeview inside the glmdir output folder.
%
% Inputs:
% fsgd: the fsgd file containing the classes, inputs, covariates, etc.
% mtx: the contrast matrix file
%
% Outputs:
% - two .mgh files produced from mris_preproc and mri_surf2surf which are
% both saved in the thickness_preproc file in $SUBJECTS_DIR
% - the specific glmdir folder which within its subfolder contains various 
% files (notably the sig.mgh file) which can be mapped onto the cortical
% surface for visualization (saved into $SUBJECTS_DIR/glmdir)


function fsgd_cross_thickness(fsgd,mtx)
    %% running mris_preproc function
    preproc_output = strcat(fsgd(1:strfind(fsgd,'.fsgd')),'thickness.stack.mgh');
    cmds = cell(1,2);
    cmds{1} = strcat('mris_preproc --fsgd $SUBJECTS_DIR/fsgd/', fsgd,...
        ' --target fsaverage --hemi lh --meas thickness --out $SUBJECTS_DIR/thickness_preproc/lh.',...
        preproc_output);
    cmds{2} = strcat('mris_preproc --fsgd $SUBJECTS_DIR/fsgd/', fsgd,...
        ' --target fsaverage --hemi rh --meas thickness --out $SUBJECTS_DIR/thickness_preproc/rh.',...
        preproc_output);
    parfor ii = 1:numel(cmds)
        system(cmds{ii})
    end
    %% running mri_surf2surf function
    cmds = cell(1,2);
    surf2surf_output = strcat(preproc_output(1:strfind(preproc_output,'.mgh')),...
        'fwhm10.mgh');
    cmds{1} = strcat('mri_surf2surf --hemi lh --s fsaverage --sval $SUBJECTS_DIR/thickness_preproc/lh.',...
        preproc_output, ' --tval $SUBJECTS_DIR/thickness_preproc/lh.',surf2surf_output, ...
        ' --fwhm-trg 10 --cortex --noreshape');
    cmds{2} = strcat('mri_surf2surf --hemi rh --s fsaverage --sval $SUBJECTS_DIR/thickness_preproc/rh.',...
        preproc_output, ' --tval $SUBJECTS_DIR/thickness_preproc/rh.',surf2surf_output, ...
        ' --fwhm-trg 10 --cortex --noreshape');
    system(cmds{1})
    system(cmds{2})
    %% running mri_glmfit function
    cmds = cell(1,2);
    glmdir = strcat(surf2surf_output(1:strfind(mtx,'.mtx')),'glmdir');
    cmds{1} = strcat('mri_glmfit --y $SUBJECTS_DIR/thickness_preproc/lh.',surf2surf_output,...
        ' --fsgd $SUBJECTS_DIR/fsgd/', fsgd,' dods --C $SUBJECTS_DIR/mtx/', mtx,...
        '  --surf fsaverage lh --cortex --glmdir $SUBJECTS_DIR/glmdir/lh.',glmdir);
    cmds{2} = strcat('mri_glmfit --y $SUBJECTS_DIR/thickness_preproc/rh.',surf2surf_output,...
        ' --fsgd $SUBJECTS_DIR/fsgd/', fsgd,' dods --C $SUBJECTS_DIR/mtx/', mtx,...
        '  --surf fsaverage rh --cortex --glmdir $SUBJECTS_DIR/glmdir/rh.',glmdir);
    system(cmds{1})
    system(cmds{2})
end