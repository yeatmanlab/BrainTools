function nf_fmri_glm(data_dir)
% fits GLM and computes contrast maps. Note: there must be the same number
% of scans and parfiles and they must be in the same order for it to work. 

% Open session
cd(data_dir)
vw = initHiddenInplane();

vw = viewSet(vw, 'current dt', 'Original');
%% Prepare scans for GLM


home = pwd;
% grab parfs and figure out how many runs there are 
cd Stimuli/parfiles
parfs = dir('*.par');

nruns = size(parfs);
nruns = nruns(1);
whichScans = 1:nruns;
whichParfs = [];
for nn = 1:nruns 
    whichParfs = [whichParfs {parfs(nn).name}];
end 

cd(home)
vw = er_assignParfilesToScans(vw, whichScans, whichParfs); % Assign parfiles to scans

dt = 'Original';
vw = er_groupScans(vw, whichScans, [], dt); % Group scans together


% Check assigned parfiles and groups
er_displayParfiles(vw);


%% run the glm
dt = 'Original';
newDtName = 'GLMs';

% GLM parameters
params = er_defaultParams;
params.detrend = 2; %changes detrend to quadtadic 
params.framePeriod = 2.2; %sets TR to 2
params.glmHRF     =  3;     % spm difference of two gamma functions


%apply GLM for grouped scans
vw = applyGlm(vw, dt, whichScans, params, newDtName);
updateGlobal(vw);
% 
% %compute VWFA contrast map
% 
% stim     = er_concatParfiles(vw);
% active   = [1 2 3 4]; % words
% control  = [5 6 7 8]; % everything else
% saveName = [];
% vw       = computeContrastMap2(vw, active, control, saveName);
% 
% 
% updateGlobal(vw);
% 
% %compute FFA contrast map
% 
% stim     = er_concatParfiles(vw);
% active   = [5 6]; % faces
% control  = [1 2 3 4 7 8]; % everything else
% saveName = [];
% vw       = computeContrastMap2(vw, active, control, saveName);


updateGlobal(vw);

