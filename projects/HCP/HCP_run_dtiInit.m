function sub_dirs = HCP_run_dtiInit(baseDir, ncores, bvalue)
% Checks to see if multiple or single subjects are being run, sets
% parameters for dtiInit, corrects data and formatting for dtiInit
% analyses, runs dtiInit for all subjects.
%
% sub_dirs = HCP_run_dtiInit(baseDir, ncores)
%
% Inputs:
% baseDir - points to a folder containing all the directories containing
% the subject MR data. Will work with complete HCP dataset, or with any
% number of extracted directories downloaded from the HCP 'Diffusion
% Prepocessed' (has not been tested on Connectome in a Box or Amazon data,
% but as file structure should be consistent across formats, should be
% compatible--confirmation feedback appreciated!)
%
% ncores - number of cores to run it on
%
% All directories in baseDir should be subject directories (6 digit ID
% folders) for proper function.
%
% Creates data.bvecs and data.bvals in 'Diffusion' directory, leaving
% original files unedited while applying minor corrections to make data
% comply with dtiInit. see 'HCP_dataPrep' for further documentation
%
% Output data structure of dtiInit described in detail at: 
% http://web.stanford.edu/group/vista/cgi-bin/wiki/index.php/DTI_Preprocessing#Setting_parameters:_dtiInitParams
% **** Jason, the present output structure of dtiInit is creating a number
% of potentially extra files, the purpose of which I'm not certain.
% Documentation isn't present on the website...
%
%
% Example:
% baseDir = '/mnt/scratch/HCP900'
% HCP_run_dtiInit(baseDir)
% 

%% Argument checking
if ~exist('ncores', 'var') || isempty(ncores)
    ncores = 6;
end

%% Autodetect all subject directories
% Returns cell vector of subject directory names in baseDir
dirList = HCP_autoDir(baseDir);

%% Set dwParams
% HCP_params sets dwParams for dtiInit to correctly handle HCP data
if ~exist('hcp_paramsSet','var') || ~hcp_paramsSet
    dwParams = HCP_params;
end

%% Run each subject

if ncores > 1
    pool = parpool(ncores);
    % This is the workhorse of the function.
    parfor ii = 1:numel(dirList)
        subjectDir = fullfile(baseDir,dirList{ii},'T1w');
        [diff, t1, InitParams] = HCP_dataPrep(subjectDir, dwParams);
        % Run dtiInit
        sub_dirs{ii} = fileparts(dtiInit(diff, t1, InitParams));
    end
else
    %% Non parallel version
    for ii = 1:numel(dirList)
        subjectDir = fullfile(baseDir,dirList{ii},'T1w');
        [diff, t1, InitParams] = HCP_dataPrep(subjectDir, dwParams);
        % Run dtiInit
        sub_dirs{ii} = fileparts(dtiInit(diff, t1, InitParams));
    end
end

return

%% run AFQ
afq = AFQ_Create('sub_dirs', sub_dirs, 'sub_group', ones(length(sub_dirs,1));
afq = AFQ_run_sge(afq,[],3);

%% to extract a single bvalue
if exist('bvalue', 'var') && ~isempty(bvalue)
    for ii = 1:length(subs)
        % To extract a single bvalue
        dtiExtractSingleShell(dMRI, bvecs, bvals, brange, outname)
    end
end