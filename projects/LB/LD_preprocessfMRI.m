function LB_preprocessfMRI(datadir)

% This is a script that calls preprocessfmri_CNI.m.
% Edit the top section of this script to fit your needs and then run it.

% history:
% 2013/03/08 - move matlabpool to the script
% 2013/03/04 - add back numepiignore
% 2013/03/04 - automate epiinplanematrixsize, epiphasedir, epireadouttime based on CNI header information
% 2013/02/27 - first version

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EDIT THIS:
% Stuff that I do not have
fieldmapB0files = []
fieldmapMAGfiles=[]
inplanefilenames=[]
fieldmaptimes=[]
fieldmapdeltate=[]
fieldmapunwrap=[]
fieldmapsmoothing=[]
epifieldmapasst=[]
% what directory do the data live in?
datadir = '/biac4/wandell/data/Lindamood_Bell/MRI/adult/RH_20130627/20130627_1154';

% where should i save figures to?
figuredir = '/biac4/wandell/data/Lindamood_Bell/MRI/adult/RH_20130627/20130627_1154/preprocess';


% what NIFTI files should we interpret as EPI runs?
epifilenames = matchfiles([datadir '/*fMRI*/*.nii*'], 'tr');

% what is the desired in-plane matrix size for the EPI data?
% this is useful for downsampling your data (in order to save memory) 
% in the case that the data were reconstructed at too high a resolution.  
% for example, if your original in-plane matrix size was 70 x 70, the 
% images might be reconstructed at 128 x 128, in which case you could 
% pass in [70 70].  what we do is to immediately downsample each slice
% using lanczos3 interpolation.  if [] or not supplied, we do nothing special.
epidesiredinplanesize = [];

% what is the slice order for the EPI runs?
% special case is [] which means to omit slice time correction.
episliceorder = 'interleaved';

% how many volumes should we ignore at the beginning of each EPI run?
numepiignore = 3;

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
motioncutoff = [];

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
mcmask =  [];

% how should we handle voxels that have NaN values after preprocessing?
% if [], we use the default behavior which is to zero out all voxels that have a NaN
% value at any point in the EPI data.  see preprocessfmri.m for other options.
maskoutnans = [];

% savefile:  what .nii files (accepting a 1-indexed integer) should we save the final EPI data to?
%            (in the special EPI flattening case, we save the data to raw binary files (time x voxels) instead of .nii files.)
% (we automatically make parent directories if necessary.)
savefile = [datadir '/run%02d.nii'];

% what .txt file should we keep a diary in?
diaryfile = [datadir '/diary.txt'];

%% Run KNK preprocess

  mkdirquiet(stripfile(diaryfile));
  diary(diaryfile);
preprocessfmri_CNI;
  diary off;

%% Copy the processed files to a raw directory
data=[];
cd(datadir)
mkdir('RAW')
for ii = 1:length(epifilenames)
    im = readFileNifti(fullfile(datadir,sprintf('run%02d.nii',ii)));
    data = cat(4,data,im.data);
    movefile(fullfile(datadir,sprintf('run%02d.nii',ii)),fullfile(datadir,'RAW',sprintf('run%02d.nii',ii)));
end
datam = nanmean(data,4);
im.data = datam;
im.pixdim = im.pixdim(1:3);
im.dim = im.dim(1:3);
im.ndim = 3;
im.descrip = 'meanfMRI';
im.fname = fullfile(datadir,'RAW','MeanFunctional.nii.gz');
writeFileNifti(im)


%% Align t1 to mean functional
f = im; % Mean functional
t1FileName = matchfiles([datadir '/*FSPGR_BRAVO*/*.nii*'], 'tr');
t1 = readFileNifti(t1FileName{1});
seed = inv(t1.qto_xyz)*f.qto_xyz;  % EPI image space to T1 image space

% get into KK format
T = matrixtotransformation(seed,0,t1.pixdim(1:3),f.dim(1:3),f.dim(1:3).*f.pixdim(1:3));

% call the alignment
alignvolumedata(double(t1.data),t1.pixdim(1:3),double(f.data),f.pixdim(1:3),T);
% Define ellipse
[~,mn,sd] = defineellipse3d(double(f.data));

%% Automatic alignment (coarse)
useMI = true;  % you need MI if the two volumes have different tissue contrast.
               % it's much faster to not use MI.
alignvolumedata_auto(mn,sd,0,[4 4 4],[],[],[],useMI);  % rigid body, coarse, mutual information metric

%% Automatic alignment (fine)
alignvolumedata_auto( mn, sd,0,[1 1 1],[],[],[],useMI);  % rigid body, fine, mutual information metric


%% Export the final transformation
tr = alignvolumedata_exporttransformation;

% make the transformation into a 4x4 matrix
T = transformationtomatrix(tr,0,t1.pixdim(1:3));

% %% (5) Save as alignment for your vista session 
% vw = initHiddenInplane; mrGlobals; 
% mrSESSION.alignment = T;
% saveSession;


%% Optional: Save images showing the alignment

t1match = extractslices(double(t1.data),t1.pixdim(1:3),double(f.data),f.pixdim(1:3),tr);

f.data = t1match;
f.fname = 'RAW/Inplane.nii.gz';
writeFileNifti(f);

% % inspect the results
% if ~exist('Images', 'dir'), mkdir('Images'); end
% imwrite(uint8(255*makeimagestack(refpre,1)),'Images/inplane.png');
% imwrite(uint8(255*makeimagestack(t1match,1)),'Images/reslicedT1.png');
% 

%% Create mrsession

params.inplane = 'RAW/Inplane.nii.gz';
for ii = 1:3; params.functionals{ii} = sprintf('RAW/run%02d.nii', ii); end
params.vAnatomy = '3DAnatomy/t1.nii.gz';
params.annotations = {'Localizer' 'Dot' 'Rhyme'};
mrInit(params)

%% Open mrAlign to get alignment structure
rxAlign; 
% Once you are done with the alignment, pull out the necessary info
rxVista = rxRefresh;
rxClose;
rx = rxVista; clear rxVista;
close all;

%% (3) get into knk format 
% (why doesn't he just take the 4x4?)
% the reason is that the coordinate-space conventions are different.
rxAlignment = rx.xform;
rxAlignment([1 2],:) = rxAlignment([2 1],:);
rxAlignment(:,[1 2]) = rxAlignment(:,[2 1]);
knk.TORIG = rxAlignment;
knk.trORIG = matrixtotransformation(knk.TORIG,0,rx.volVoxelSize,size(rx.ref),size(rx.ref) .* rx.refVoxelSize);
%% (3) get into knk format 
% (why doesn't he just take the 4x4?)
% the reason is that the coordinate-space conventions are different.
rxAlignment = rx.xform;
rxAlignment([1 2],:) = rxAlignment([2 1],:);
rxAlignment(:,[1 2]) = rxAlignment(:,[2 1]);
knk.TORIG = rxAlignment;
knk.trORIG = matrixtotransformation(knk.TORIG,0,rx.volVoxelSize,size(rx.ref),size(rx.ref) .* rx.refVoxelSize);


alignvolumedata(volpre,rx.volVoxelSize,refpre,rx.refVoxelSize,knk.trORIG);

%% 4c Automatic alignment (coarse)
useMI = false;  % you need MI if the two volumes have different tissue contrast.
               % it's much faster to not use MI.
alignvolumedata_auto([],[],0,[4 4 2],[],[],[],useMI);  % rigid body, coarse, mutual information metric

%% 4d Automatic alignment (fine)
alignvolumedata_auto(mn,sd,0,[1 1 1],[],[],[],useMI);  % rigid body, fine, mutual information metric

%% 4e Export the final transformation
tr = alignvolumedata_exporttransformation;

% make the transformation into a 4x4 matrix
T = transformationtomatrix(tr,0,rx.volVoxelSize);

%% (5) Save as alignment for your vista session 
vw = initHiddenInplane; mrGlobals; 
mrSESSION.alignment = T;
saveSession;

close all
