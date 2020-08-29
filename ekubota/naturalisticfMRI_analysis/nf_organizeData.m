function nf_organizeData(sub_list,session)
% Checks raw folders for fMRI data and then creates an fMRI folder for
% analysis 


% Get subject numbers for everyone in PREK_Analysis folder 
root_dir = '/mnt/scratch/PREK_Analysis/';

    % For each subject (si = subject index)
    for si = 1:length(sub_list)
        raw_dir = strcat(root_dir, sub_list{si}, '/',session,'/raw');
        cd(raw_dir);
        EPI_names = dir('*fMRI*.PAR'); % check the raw directory for 70V functionals
        nFunctionals = size(EPI_names);
        nFunctionals = nFunctionals(1);
        if ~isempty(EPI_names) %If there are functionals, make fmri directory and convert parrec files
            cd ..
            mkdir('func') %Make fmri folder
            cd func
            mkdir('Stimuli/parfiles')
            funDir = strcat(root_dir,'/',sub_list{si},'/',session,'/func/');
            cd(raw_dir)
            % Convert to nifti and sace into functional directory
            for fi = 1:nFunctionals
                parrecCommand = strcat('parrec2nii -o', {' '},funDir,{' '},'-c -b',{' '},EPI_names(fi).name);
                parrecCommand = parrecCommand{1};
                system(parrecCommand)
            end
            
            % Check to see if we have anatomy folder, and if not make one
            sub_folder = strcat(root_dir, '/',sub_list{si},'/',session);
            %sub_folder = sub_folder{1};
            cd(sub_folder)
            anatFolderCheck = exist(fullfile(sub_folder,'t1'),'dir');
            if anatFolderCheck == 0
                mkdir('t1')
            end
        end
    end
end

