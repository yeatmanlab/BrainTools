function HCP_singleRun(basedir)
% BDE lab preprocessing for HCP data
% 
% Esoteric to dtiInit processing of files unzipped from 'Diffusion
% Preprocessed' downloaded from db.humanconnectome.org
%
% Sets up preprocessed HCP data for dtiInit
%
%
% Will rewrite in future to accommodate automatic processing of many
% subject files, for now, set up to process one at a time.
%
%
% 'basedir' should be entered as the T1w level directory unzipped from 
% '*_3T_Diffusion_preproc.zip'
%
% Example: 
%
% basedir = '/home/user/Documents/HCP/103414/T1w'
% HCP_dataPrep(basedir)

%% Set dwParams

if ~exist('hcp_paramsSet','var') || ~hcp_paramsSet
    HCP_params;
end


%% Set up directories and find files
diffdir = fullfile(basedir, 'Diffusion');
t1 = fullfile(basedir, 'T1w_acpc_dc_restore_1.25.nii.gz');
bvalsR = fullfile(diffdir, 'bvals');
bvecsR = fullfile(diffdir, 'bvecs');
diff = fullfile(diffdir, 'data.nii.gz'); 

% Read in bvecs file, flip over X axis, write in correct format

% Load in bvecs text file
bve_xflip  = dlmread(bvecsR);

% Flip over x axis
bve_xflip(1,:) = bve_xflip(1,:).*-1;

% Write the file out with new formatting
dlmwrite(fullfile(diffdir,'data.bvecs'), bve_xflip, '\t')

% % Define fixed file
% bvecsF = fullfile(diffdir,'data.bvecs');


% Read in a .bvals file and round bvalues, write in correct format

% Load in bvals text file
b = dlmread(bvalsR);

% We're only dealing with the b=0 volumes for now
b(b<10) = 0;

% write the file out with new formatting
dlmwrite(fullfile(diffdir,'data.bvals'),b,'\t')

% % Define fixed file
% bvalsF = fullfile(diffdir,'data.bvals');


%% Define bvals and bvecs and run dtiInit

dwParams.bvecsFile = fullfile(diffdir,'data.bvecs');
dwParams.bvalsFile = fullfile(diffdir,'data.bvals');

dtiInit(diff, t1, dwParams)


