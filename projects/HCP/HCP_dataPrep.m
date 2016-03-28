function HCP_dataPrep(subjectDir, dwParams)
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
% HCP_dataPrep(basedir, dwParams)

%% Set up directories and find files
diffDir = fullfile(subjectDir, 'Diffusion');
t1 = fullfile(subjectDir, 'T1w_acpc_dc_restore_1.25.nii.gz'); % in future, may point to .7mm T1 data
bvalsR = fullfile(diffDir, 'bvals');
bvecsR = fullfile(diffDir, 'bvecs');
diff = fullfile(diffDir, 'data.nii.gz'); 

% Read in bvecs file, flip over X axis, write in correct format

% Load in bvecs text file
bve_xflip  = dlmread(bvecsR);

% Flip over x axis
bve_xflip(1,:) = bve_xflip(1,:).*-1;

% Write the file out with new formatting
dlmwrite(fullfile(diffDir,'data.bvecs'), bve_xflip, '\t')


% Read in a .bvals file and round bvalues, write in correct format
% bvals below 10 are rounded to 0, as b=0 necessary for dtiInit

% Load in bvals text file
b = dlmread(bvalsR);

% We're only dealing with the b=0 volumes for now
b(b<10) = 0;

% write the file out with new formatting
dlmwrite(fullfile(diffDir,'data.bvals'),b,'\t')


%% Define bvals and bvecs and run dtiInit

dwParams.bvecsFile = fullfile(diffDir,'data.bvecs');
dwParams.bvalsFile = fullfile(diffDir,'data.bvals');

dtiInit(diff, t1, dwParams)


