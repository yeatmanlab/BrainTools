cd /biac4/wandell/biac3/wandell7/data/Retinotopy/Kiani/RK20130317_SaccadeMislocalization
params.inplane = 'RAW/4_1_T1_Inplane_36_slices/4159_4_1.nii.gz';

for ii = 1:8; params.functionals{ii} = sprintf('RAW/run%02d.nii', ii); end

params.vAnatomy = '3DAnatomy/t1_0.7mm.nii.gz';


params.annotations = {...
    'saccade + probe' ...   scan 1
    'saccade + probe' ...   scan 2
    'fixation' ...          scan 3
    'saccade + probe' ...   scan 4
    'saccade + probe' ...   scan 5
    'saccade + probe' ...   scan 6
    'saccade + probe' ...   scan 7
    'saccade + probe'}; %   scan 8
mrInit(params)

% Scan 1: saccade + probe
% Scan 2: saccade
% Scan 3: fixation
% Scan 4: saccade + probe
% Scan 5: saccade + probe
% Scan 6: saccade + probe
% Scan 7: saccade + probe
% Scan 8: saccade + probe