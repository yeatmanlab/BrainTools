function [diff, t1, subParams] = HCP_dataPrep(subjectDir, dwParams, shell)
% BDE lab preprocessing for HCP data
% 
% Esoteric to dtiInit processing of files unzipped from 'Diffusion
% Preprocessed' downloaded from db.humanconnectome.org
%
% HCP_run_dtiInit will fully automate this entire process and requires only
% the entry of the directory containing all subjects you wish to run.
%
% CAN be run indepedently, if so:
% Sets up preprocessed HCP data for and executes dtiInit. 
%
% dwParams can be set manually, or automatically using HCP_params
%
% 'subjectDir' should be entered as the T1w level directory unzipped from 
% '*_3T_Diffusion_preproc.zip'
%
% Example: 
%
% subjectDir = '/home/user/Documents/HCP/103414/T1w';
% dwParams = HCP_params;
% [diff, t1, subParams] = HCP_dataPrep(basedir, dwParams, shell)

%% Define shell parameters
% Set range for b value for selected shell
if shell == 1
    bRange = [900 1100];
    outname = 'shell1';
elseif shell == 2
    bRange = [1900 2100];
    outname = 'shell2';
elseif shell == 3
    bRange = [2900 3100];
    outname = 'shell3';
end


%% Set up directories and find files
diffDir = fullfile(subjectDir, 'Diffusion');
t1 = fullfile(subjectDir, 'T1w_acpc_dc_restore_1.25.nii.gz'); % in future, may point to .7mm T1 data
b = fullfile(diffDir, 'bvals');
bv = fullfile(diffDir, 'bvecs');
diff = fullfile(diffDir, 'data.nii.gz'); 

% Load in T1 nifti, bvals, & bvecs
diff = readFileNifti(diff);
b = dlmread(b);
bv  = dlmread(bv);

% bvals below 10 are rounded to 0, as b=0 necessary for dtiInit
b(b<10) = 0;

% Flip bvecs over x axis to correct orientation
bv(1,:) = bv(1,:).*-1;

% Reorientate bvecs and bvals for operation (necessary?)
% if size(bv,2) > size(bv,1)
%     bv = bv';
% end
% if size(b,2) > size(b,1)
%     b = b';
% end

%% Find dMRI volumes with bvalues in the specified range (or equal to zero)
v = (b > bRange(1) & b < bRange(2)) | b == 0;
% Extract image volumes and corresponding bvals
diff.data = diff.data(:,:,:,v); diff.dim(4)=sum(v); diff.fname = [outname '.nii.gz'];
b = b(v); bv = bv(v,:);

% Write the bvecs out with new formatting
dlmwrite(fullfile(diffDir, [outname '.bvecs']), bv, '\t')

% write the bvals out with new formatting
dlmwrite(fullfile(diffDir, [outname '.bvals']),b,'\t')


%% Define subParams, define bvals and bvecs

subParams = dwParams;

subParams.bvecsFile = fullfile(diffDir,'data.bvecs');
subParams.bvalsFile = fullfile(diffDir,'data.bvals');




