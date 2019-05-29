function nf_organizeData
% Checks raw folders for fMRI data and then creates an fMRI folder for
% analysis 

sub_list = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};

% Get subject numbers for everyone in PREK_Analysis folder 
root_dir = '/mnt/scratch/PREK_Analysis/';

    % For each subject (si = subject index)
for si = 1:length(sub_list)
    % Get the visit dates 
    visit_dir = strcat(root_dir,'/',sub_list{si});
    visit_dates = HCP_autoDir(visit_dir);
    % For each visit (vi = visit index)
    for vi = 1:length(visit_dates)
        % Check to see if the vist date folder is actually a date
        a = visit_dates{vi};
        sizeA = size(a);
        sizeA = sizeA(2);
        if sizeA == 8
            raw_dir = strcat(root_dir, sub_list{si}, '/',visit_dates{vi},'/raw');
            cd(raw_dir);
            EPI_names = dir('*fMRI*.PAR'); % check the raw directory for 70V functionals 
            nFunctionals = size(EPI_names);
            nFunctionals = nFunctionals(1);
            if ~isempty(EPI_names) %If there are functionals, make fmri directory and convert parrec files
                cd .. 
                mkdir('fmri') %Make fmri folder
                cd fmri
                mkdir('Stimuli/parfiles')
                funDir = strcat(root_dir,'/',sub_list{si},'/', visit_dates{vi},...
                '/', 'fmri');
                cd(raw_dir)
            % Convert to nifti and sace into functional directory
                for fi = 1:nFunctionals
                    parrecCommand = strcat('parrec2nii -o', {' '},funDir,{' '},'-c -b',{' '},EPI_names(fi).name);
                    parrecCommand = parrecCommand{1};
                    system(parrecCommand)           
                end
            
            % Check to see if we have anatomy folder, and if not make one
                sub_folder = strcat(root_dir, '/',sub_list{si});
                %sub_folder = sub_folder{1};
                cd(sub_folder)
                anatFolderCheck = exist(fullfile(sub_folder,'nf_anatomy'),'dir');
                if anatFolderCheck == 0
                    mkdir('nf_anatomy')
                end 
            end
        end 
    end 
end
            