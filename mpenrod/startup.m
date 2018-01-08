%------------ FreeSurfer -----------------------------%
fshome = getenv('FREESURFER_HOME');
fsmatlab = sprintf('%s/matlab',fshome);
if (exist(fsmatlab) == 7)
    addpath(genpath(fsmatlab));
end
clear fshome fsmatlab;
%-----------------------------------------------------%

%------------ FreeSurfer FAST ------------------------%
fsfasthome = getenv('FSFAST_HOME');
fsfasttoolbox = sprintf('%s/toolbox',fsfasthome);
if (exist(fsfasttoolbox) == 7)
    path(path,fsfasttoolbox);
end
clear fsfasthome fsfasttoolbox;
%-----------------------------------------------------%


% other bde lab code paths
addpath(genpath('~/git/yeatmanlab'));
addpath(genpath('~/Documents/MATLAB/spm8/'))
addpath(genpath('~/git/AFQ/'))
addpath(genpath('~/git/BrainTools/mritools'))
addpath(genpath('~/git/vistasoft/'))
addpath(genpath('~/matlab/'))
addpath(genpath('/mnt/scratch/projects/freesurfer/'))
addpath(genpath('/usr/local/freesurfer'))
