% Script which sifts through the data returning all subjects and sessions
% which have the appropriate 'VBM' files 

%% setup directory info
subID = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH',...
    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_127_AM', ...
    'NLR_105_BB', 'NLR_132_WP', 'NLR_187_NB', 'RI_124_AT', 'RI_143_CH', ...
    'RI_138_LA', 'RI_141_GC', 'RI_144_OL','NLR_199_AM', 'NLR_130_RW', ...
    'NLR_133_ML', 'NLR_146_TF', 'NLR_195_AW', 'NLR_191_DF', 'NLR_197_BK'};
maindir = '/mnt/scratch/MRI/';
anatdir = '/mnt/scratch/anatomy/';
addpath(genpath('/mnt/scratch'))

%%
anat_check = cell(numel(subID)+1,6);
anat_check(1,:) = {'Subject','Session 1', 'Session 2', 'Session 3', 'Session 4', 'Follow up'};
avg_ref_check = cell(numel(subID)+1,2);
avg_ref_check(1,:) = {'Subject','Avg File?'};
for ss = 1:numel(subID)
    subject = subID{ss};
    % Find the folders labeled by date and pick desired session:
    allsessions = dir(fullfile(maindir, subject));
    %   (session folders have form yyyy/mm/dd, so length is 8 chars)
    allsessions = allsessions(cellfun(@length, {allsessions.name})==8);
    anat_check(ss+1,1) = {strcat(subID{ss},'')};
    avg_ref_check(ss+1,1) = {strcat(subID{ss},'')};
    for session = 1:numel(allsessions)
        for qq = numel(allsessions)+2:6
            anat_check(ss+1,qq) = {'N/A'};
        end
        
        sessiondir = allsessions(session).name;
        
        rawdir = fullfile(maindir, subject, sessiondir, 'raw');
        cd(rawdir)
        
        T1path = dir(fullfile(rawdir, '*VBM*.nii.gz'));
        
        % Convert PAR/REC files to make nifti if it doesn't exist yet
        if isempty(T1path)
            cmd = sprintf('parrec2nii -b -c --scaling=%s --store-header --output-dir=%s --overwrite %s', ...
                'dv', rawdir, '*VBM*.PAR');
            system(cmd) % convert_parrec(cellstr(parfiles), rawdir);
            T1path = dir(fullfile(rawdir, '*VBM*.nii.gz'));
        end
        % Add session to anat_check otherwise note missing anatomy files
        if(~isempty(T1path));
            anat_check(ss+1,session+1) = {sessiondir};
        else
            anat_check(ss+1,session+1) = {'Missing anatomy'};
        end
        
        avg_file = fullfile(anatdir,subject,'t1_acpc_avg.nii.gz');
        if exist(avg_file, 'file')
            avg_ref_check(ss+1,2) = {'yes'};
        else
            avg_ref_check(ss+1,2) ={'no'};
        end
    end
end
