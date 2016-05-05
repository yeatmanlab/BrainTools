function afq = HCP_AFQonly(baseDir)
% Checks to see if multiple or single subjects are being run, sets
% parameters for dtiInit, corrects data and formatting for dtiInit
% analyses, runs dtiInit for all subjects.
%
% Tensor model fitting only appropriate for a single shell. As HCP
% data has three shells, call requires specification of shell to use.
%
% sub_dirs = HCP_run_dtiInit(baseDir, numCores, numShells)
%
% Inputs:
% baseDir - points to a folder containing all the directories containing
% the subject MR data. Will work with complete HCP dataset, or with any
% number of extracted directories downloaded from the HCP 'Diffusion
% Prepocessed' (has not been tested on Connectome in a Box or Amazon data,
% but as file structure should be consistent across formats, should be
% compatible--confirmation feedback appreciated!)
%
% % numCores - number of cores to run it on
%
% shell - Shell to be analyzed. Defaults to inner shell (1). HCP
% data has 3 shells, with approximate bvals of 1000, 2000, and 3000; 0 is
% reference and included by default)
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
% numCores = 6
% numShells = 1
% sub_dirs = HCP_run_dtiInit(baseDir, numCores, shell)
% 

%% Clock for testing
tic


%% Autodetect all subject directories
% Returns cell vector of subject directory names in baseDir
dirList = HCP_autoDir(baseDir);


%% Set sub_dirs for speed
sub_dirs = cell(1, numel(dirList));

%% Run each subject
% This is the workhorse of the function.

% In parallel
% if numCores > 1
for ii = 1:numel(dirList)
    temp = dir(horzcat(baseDir, '/', dirList{ii}, '/T1w/dti*trilin'));
    sub_dirs{ii} = horzcat(char(baseDir), '/', char(dirList{ii}), '/T1w/', temp.name);
end
% else
%     % Non parallel version
%     for ii = 1:numel(dirList)
%         subjectDir = fullfile(baseDir,dirList{ii},'T1w');
%         [diff, t1, subParams] = HCP_dataPrep(subjectDir, dwParams, shell);
%         % Run dtiInit, record file outputs and return
%         sub_dirs{ii}([dt6FileName, outBaseDir]) = dtiInit(diff, t1, subParams);
%     end
% end

tic
%% run AFQ
afq = AFQ_Create('sub_dirs', sub_dirs, 'sub_group', ones(length(sub_dirs),1), 'sub_names', dirList, 'seedVoxelOffsets', .5, 'clip2rois', 0);
afq = AFQ_run_sge(afq,[],3);

%% Clock for testing
toc

