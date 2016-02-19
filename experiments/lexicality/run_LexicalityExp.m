function run_LexicalityExp(runnum, stimfile)
% Run the lexicality experiment with ptviewmovie

if ~exist('runnum','var') || isempty(runnum)
    runnum = 1;
end
if ~exist('stimfile','var') || isempty(stimfile)
    path = fileparts(which('run_LexicalityExp.m'));
    stimfile = fullfile(path,'LexicalityExp.mat');
end
% Load the stimulus file
load(stimfile);

fprintf('\n\nRUNNING LEXICALITY EXPERIMENT STIMFILE %s\nRUN %d',stimfile,runnum);
%% Set experiment parameters
runnum = 1; %First run.
skipsync = 1
offset = [];  % [] means no translation of the stimuli
movieflip = [0 0];  % [0 0] means no flips.  [1 0] is necessary for flexi mirror to show up right-side up
frameduration = 12;  % number of monitor frames for one unit.  60/5 = 12
ptonparams = {[],[],0,skipsync};  % don't change resolution

% Size of fixation
fixationsize = [8 0];
grayval = uint8(127);
scfactor = 1;  % scale images bigger or smaller
%tfun = [];

%% Run experiment
oldclut = pton(ptonparams{:});
[timeframes,timekeys,digitrecord,trialoffsets] = ...
    ptviewmovie(reshape(img,[size(img,1), size(img,2), 1 , size(img,3)]), ...
    frameorder(runnum,:),[],frameduration,fixorder,fixcolor, ...
    fixationsize,grayval,[],[],offset,[],movieflip,scfactor,[], ...
    [],[],[],'5',[],[]);
ptoff(oldclut);
