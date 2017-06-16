% Script which runs through all the sessions for the subjects and makes
% sure all sessions were acpc preprocessed

% Set up directory info
%%
subID = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH',...
    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_127_AM', ...
    'NLR_105_BB', 'NLR_132_WP', 'NLR_187_NB', 'RI_124_AT', 'RI_143_CH', ...
    'RI_138_LA', 'RI_141_GC', 'RI_144_OL','NLR_199_AM', 'NLR_130_RW', ...
    'NLR_133_ML', 'NLR_146_TF', 'NLR_195_AW', 'NLR_191_DF', 'NLR_197_BK'};
clobber = 1;
makesurface = 0;
maindir = '/mnt/scratch/MRI/';
anatdir = '/mnt/scratch/anatomy/';
addpath(genpath('/mnt/scratch'))
%%
missing_sessions = {};
for ss = 1:numel(subID)
    subject = subID{ss};
    % Find the folders labeled by date and pick desired session:
    allsessions = dir(fullfile(maindir, subject));
    %   (session folders have form yyyy/mm/dd, so length is 8 chars)
    allsessions = allsessions(cellfun(@length, {allsessions.name})==8);
    
    % Counts the number of sessions per subject
    session_count = 0;
    for session = 1:numel(allsessions)
        session_count = session_count + 1;
    end
    
    % Counts the number of processed files per subject
    process_count = 0;
    for processed = 1:5
        if exist(fullfile(anatdir, subject, (strcat('t1_acpc_', num2str(processed), '.nii.gz'))),'file')
            process_count = process_count +1;
        else
            break
        end
    end
    
    % Record all subjects for which not all sessions were processed
    if process_count ~= session_count
        missing_sessions = [missing_sessions, {subject}];
    end
end
disp(missing_sessions);

