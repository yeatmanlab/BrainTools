% Function which will run the mri_anatomicalstats function for multiple
% subjects/labels sequentially and without repeated entering of the command
%
% Inputs: 
% label_files: An array containg the names of the files which denote the
% ROI from which data will be gathered
% subjects: A list of the subjects (without any denotation of session)
% ROI_label: The name of the ROI (for file naming purposes)
% hemi: hemisphere of interest (also for labeling purposes)
% sessions: the sessions for which data will be collected
%
% Ouputs: 
% A series of files containing the information for each subject at each
% session. Files are saved to
% $SUBJECTS_DIR/ROI_stats/<ROI_label>_stats/<hemi>/ 
% All files are saved separately, but the function consol_ROI_stats_files
% can be used to consolidate these files into a table.
%
% Author: Mark Penrod
% Date: July 2017


function collect_ROI_stats(label_files,subjects,ROI_label,hemi,sessions)
for ii = 1:numel(label_files)
    for jj = 1:numel(subjects)
        for kk = 1:numel(sessions)
            current_sub = strcat(subjects{jj},'_',num2str(sessions(kk)),'.long.',subjects{jj},'_template');
            if exist(current_sub,'dir')
                ROI = fullfile(current_sub,'label',label_files{ii});
                if exist(ROI,'file')
                    system(strcat('mris_anatomical_stats -l',[' ',ROI],' -f $SUBJECTS_DIR/ROI_stats/',...
                        ROI_label,'_stats/',hemi,'/',current_sub,'.',label_files{ii},'.stats',[' ',current_sub],[' ',hemi]));
                end
            else 
                continue
            end
        end
    end
end             
end

