function nf_preprocessfmri(datadir)
% This script is taken from the Winawer lab wiki and is modified to
% preprocess child fmri data from the yeatman lab. 
% A number of variables at the top of the script need to be defined
% with session-specific information. 
%
% In order to run this script, MATLAB should be opened via the terminal so
% that Matlab will have access to FSL functions. 
%
% Dependencies
%   vistasoft
%   knkutils
%   winawerlabUtils
%   preprocessfmri
%   FSL

% history:
% 2013/03/08 - move matlabpool to the script
% 2013/03/04 - add back numepiignore
% 2013/03/04 - automate epiinplanematrixsize, epiphasedir, epireadouttime based on CNI header information
% 2013/02/27 - first version
% ------------------------------------------------------------------------


%% Standardize nifti files from CBI and Siemens scans


%% Specify field map information and files

% where should i save figures to?
figuredir = fullfile(datadir,'figures');
cd(datadir)

% what NIFTI files should we interpret as in-plane runs?
inplanefilenames = matchfiles(fullfile(datadir, '*Inplane*.nii*'),'tr');

% what NIFTI files should we interpret as EPI runs?
epifilenames = matchfiles(fullfile(datadir, '*fMRI*.nii*'));
disp(epifilenames');


epiTimeFactor = 1; %  convert from ms to seconds b/c CBI tr is in ms and code assumes seconds

% what is the desired in-plane matrix size for the EPI data?
% this is useful for downsampling your data (in order to save memory) 
% in the case that the data were reconstructed at too high a resolution.  
% for example, if your original in-plane matrix size was 70 x 70, the 
% images might be reconstructed at 128 x 128, in which case you could 
% pass in [70 70].  what we do is to immediately downsample each slice
% using lanczos3 interpolation.  if [] or not supplied, we do nothing special.
epidesiredinplanesize = [80 80];

% what is the slice order for the EPI runs?
% special case is [] which means to omit slice time correction.
%   If even number of slices, then [2:2:end 1:2:end]
%   If odd number of slices, then [1:2:end 2:2:end]
episliceorder = 1:33; % for our experiment ascending -EK



%% Speficy field map correction parameters
% how many volumes should we ignore at the beginning of each EPI run?
numepiignore = 0;

% what volume should we use as reference in motion correction? ([] indicates default behavior which is
% to use the first volume of the first run; see preprocessfmri.m for details.  set to NaN if you
% want to omit motion correction.)
motionreference = [];

% for which volumes should we ignore the motion parameter estimates?  this should be a cell vector
% of the same length as the number of runs.  each element should be a vector of indices, referring
% to the volumes (after dropping volumes according to <numepiignore>).  can also be a single vector
% of indices, in which case we use that for all runs.  for volumes for which we ignore the motion
% parameter estimates, we automatically inherit the motion parameter estimates of the closest
% volumes (if there is a tie, we just take the mean).  [] indicates default behavior which is to 
% do nothing special.
epiignoremcvol = [];

% by default, we tend to use double format for computation.  but if memory is an issue,
% you can try setting <dformat> to 'single', and this may reduce memory usage.
dformat = 'double';

% what cut-off frequency should we use for filtering motion parameter estimates? ([] indicates default behavior
% which is to low-pass filter at 1/90 Hz; see preprocessfmri.m for details.)
motioncutoff = Inf; % No low pass filter for the kids -ECK

% what extra transformation should we use in the final resampling step? ([] indicates do not perform an extra transformation.)
extratrans = [];

% what is the desired resolution for the resampled volumes? ([] indicates to just use the original EPI resolution.)
targetres = [];

% should we perform slice shifting?  if so, specify band-pass filtering cutoffs in Hz, like [1/360 1/20].
% probably should be left as [] which means to do nothing special.
sliceshiftband = [];

% these are constants that are used in fmriquality.m.  it is probably 
% fine to leave this as [], which means to use default values.
% NaN means to skip the fmriquality calculations.
fmriqualityparams = [];

% what kind of time interpolation should we use on the fieldmaps (if applicable)?
% ([] indicates to use the default, which is cubic interpolation.)
fieldmaptimeinterp = [];

% should we use a binary 3D ellipse mask in the motion parameter estimation?
% if [], do nothing special (i.e. do not use a mask).
% if {}, then we will prompt the user to interactively determine the
%   3D ellipse mask (see defineellipse3d.m for details).  upon completion,
%   the parameters will be reported to the command window so that you can
%   simply supply those parameters if you run again (so as to avoid user interaction).
% if {MN SD}, then these will be the parameters that determine the mask to be used.
mcmask = [];



% how should we handle voxels that have NaN values after preprocessing?
% if [], we use the default behavior which is to zero out all voxels that have a NaN
% value at any point in the EPI data.  see preprocessfmri.m for other options.
maskoutnans = [];

%% Set save files
% savefile:  what .nii files (accepting a 1-indexed integer) should we save the final EPI data to?
% (we automatically make parent directories if necessary, and we also create a mean.nii file
% with the mean volume and a valid.nii file with a binary mask of the valid voxels.)
savefile = [datadir '/run%02d.nii'];

% what .txt file should we keep a diary in?
diaryfile = [datadir '/diary.txt'];

%% RUN PREPROCESSING FUNCTIONS
mkdirquiet(stripfile(diaryfile));
diary(diaryfile);
preprocessfmri_CBI;
diary off;