function bde_preprocessdiffusion(basedir, t1dir)
% BDE lab preprocessing for diffusion data
%
% This is essoteric to our current diffusion acquisition which includes a
% 32 direction acquisition (*DWI64_*.nii.gz) and a 64 direction acquisition
% (*DWI64_*.nii.gz), and B0 images with a reversed phase encode
% (*DWI6_*.nii.gz)
% Example:
%
% basedir = '/mnt/diskArray/projects/MRI/NLR_204_AM'
% t1dir = '/mnt/diskArray/projects/anatomy/NLR_204_AM'
% bde_preprocessdiffusion(basedir, t1dir)

%% Set up directories and find files
rawdir = fullfile(basedir,'raw');
d64 = dir(fullfile(rawdir,'*DWI64_*.nii.gz'));
d32 = dir(fullfile(rawdir,'*DWI32_*.nii.gz'));
b0 = dir(fullfile(rawdir,'*DWI6_*.nii.gz'));
dMRI64Files{1}=fullfile(rawdir,d64.name);
dMRI32Files{1}=fullfile(rawdir,d32.name);

% Add the b0 with the reversed phase encode
for ii = 1:length(b0)
    dMRI64Files{1+ii}=fullfile(rawdir,b0(ii).name);
    dMRI32Files{1+ii}=fullfile(rawdir,b0(ii).name);
end

% Bvals and Bvecs files
for ii = 1:length(dMRI64Files)
    bvals64{ii} = [prefix(prefix(dMRI64Files{ii})) '.bvals'];
    bvecs64{ii} = [prefix(prefix(dMRI64Files{ii})) '.bvecs'];
    bvals32{ii} = [prefix(prefix(dMRI32Files{ii})) '.bvals'];
    bvecs32{ii} = [prefix(prefix(dMRI32Files{ii})) '.bvecs'];
end

% Phase encode matrix. This denotes, for each volume, which direction is
% the phase encode
%pe_mat = [0 1 0; 0 1 0; 0 -1 0];
pe_mat = [0 1 0;0 -1 0];

% Directory to save everything
outdir64 = fullfile(basedir,'dmri64');
outdir32 = fullfile(basedir,'dmri32');

%% Pre process: This is mostly done with command line calls to FSL
fsl_preprocess(dMRI64Files, bvecs64, bvals64, pe_mat, outdir64);
fsl_preprocess(dMRI32Files, bvecs32, bvals32, pe_mat, outdir32);

%% Run dtiInit to fit tensor model
% Now run dtiInit. We turn off motion and eddy current correction because
% that was taken care of by FSL

% First for 64 dir data
params = dtiInitParams; % Set up parameters for controlling dtiInit
dtEddy = fullfile(outdir64,'eddy','data.nii.gz'); % Path to the data
params.bvalsFile = fullfile(outdir64,'eddy','bvals'); % Path to bvals
params.bvecsFile = fullfile(outdir64,'eddy','bvecs'); % Path to the bvecs
params.eddyCorrect=-1; % This turns off eddy current and motion correction
%params.outDir = fullfile(basedir,'dti64');
params.rotateBvecsWithCanXform=1; % Phillips data requires this to be 1
params.phaseEncodeDir=2; % AP phase encode
params.clobber=1; % Overwrite anything previously done
t1 = fullfile(t1dir,'t1_acpc.nii.gz'); % Path to the t1-weighted image
dt6FileName{1} = dtiInit(dtEddy,t1,params); % Run dtiInit to preprocess data

% Then for 32 dir data
dtEddy = fullfile(outdir32,'eddy','data.nii.gz'); % Path to the data
params.bvalsFile = fullfile(outdir32,'eddy','bvals'); % Path to bvals
params.bvecsFile = fullfile(outdir32,'eddy','bvecs'); % Path to the bvecs
dt6FileName{2} = dtiInit(dtEddy,t1,params); % Run dtiInit to preprocess data

%% Run AFQ

% Cell array with paths to the dt6 directories
dt6dirs = horzcat(fileparts(dt6FileName{1}), fileparts(dt6FileName{2}));
afq = AFQ_Create('sub_dirs',dt6dirs,'sub_group',[0 0],'clip2rois',0);
% To run AFQ in test mode so it will go quickly
% afq = AFQ_Create('sub_dirs',dt6dirs,'sub_group',[0 0],'run_mode','test');

% To run AFQ using mrtrix for tractography
% afq = AFQ_Create('sub_dirs',fileparts(dt6FileName{1}),'sub_group',0,'computeCSD',1);
afq = AFQ_run([],[],afq);


