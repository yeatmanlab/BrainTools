function nf_organizeLongitudinalData(sublist)
% Checks raw folders for fMRI data and then creates an fMRI folder for
% analysis

sessions = {'ses-pre','ses-post'};
% Get subject numbers for everyone in PREK_Analysis folder
root_dir = '/mnt/scratch/PREK_Analysis/';

% For each subject (si = subject index)
for si = 2:length(sublist)
    longdir = fullfile(root_dir,sublist{si},'longitudinal');
    mkdir(fullfile(longdir,'Stimuli/parfiles'));
    for vi = 1:length(sessions)
        raw_dir = strcat(root_dir, sublist{si}, '/',sessions{vi},'/raw');
        cd(raw_dir);
        EPI_names = dir('*fMRI*.PAR'); % check the raw directory for 70V functionals
        nFunctionals = size(EPI_names);
        nFunctionals = nFunctionals(1);
        if ~isempty(EPI_names) %If there are functionals, make fmri directory and convert parrec files
            cd(raw_dir)
            % Convert to nifti and sace into functional directory
            for fi = 1:nFunctionals
                parrecCommand = strcat('parrec2nii -o', {' '},longdir,{' '},'-c -b',{' '},EPI_names(fi).name);
                parrecCommand = parrecCommand{1};
                system(parrecCommand)
            end
        end
    end
end


