function fsl_register2freesurfer(fslbase,fsdir)

cd(fslbase)
dtidir = fullfile(fslbase,'dtifit')
%% Register to freesurfer data
if ~exist('fsdir','var') || isempty(fsdir)
    error('Please specify freesurfer directory')
end

% Make a nifti of the extracted brain
fsmridir = fullfile(fsdir,'mri');
niidir   = fullfile(fsmridir,'nifti');
if ~exist(niidir,'dir')
    mkdir(niidir);
end
bmgz = fullfile(fsmridir,'brain.mgz');
bnii = fullfile(niidir,'brain.nii.gz');
if ~exist(bnii,'file')
    cmd = sprintf('mri_convert %s %s',bmgz,bnii);
    system(cmd);
end

% convert white matter surfaces to asc
sdir = fullfile(fsdir,'surf');
lasc = fullfile(sdir,'lh.white.gii');
rasc = fullfile(sdir,'rh.white.gii');
if ~exist(lasc,'file') || ~exist(rasc,'file')
    lh = fullfile(sdir,'lh.white');
    rh = fullfile(sdir,'rh.white');
    system(sprintf('mris_convert %s %s',lh, lasc));
    system(sprintf('mris_convert %s %s',rh, rasc));
end

% Compute the transfor from the structural to the surface
cmd = sprintf('tkregister2 --mov %s/orig.mgz --targ %s/rawavg.mgz --regheader --reg junk --fslregout %s/struct2freesurfer.mat --noedit',fsmridir,fsmridir,fsmridir);
system(cmd)
% register FA to freesurfer brain extracted structural
fanii = fullfile(dtidir,'dti_FA.nii.gz');
cmd = sprintf('flirt -in %s -ref %s -omat %s/fa2struct.mat',fanii,bnii,dtidir);
system(cmd)
cmd =sprintf('convert_xfm -omat %s/struct2fa.mat -inverse %s/fa2struct.mat',dtidir,dtidir);
system(cmd);
% Concatenate registractions ---edit
cmd = sprintf('convert_xfm -omat %s/fa2freesurfer.mat -concat %s/struct2freesurfer.mat %s/fa2struct.mat',dtidir,fsmridir,dtidir);
system(cmd);
cmd = sprintf('convert_xfm -omat %s/freesurfer2fa.mat -inverse %s/fa2freesurfer.mat',dtidir,dtidir);
system(cmd)


return

