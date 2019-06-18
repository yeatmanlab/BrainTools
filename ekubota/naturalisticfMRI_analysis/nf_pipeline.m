sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};
visit_dates = {'20190525','20190525','20190525','20190525','20190525',...
    '20190524','20190524','20190525','20190525'};
% nf_organizeData(sublist)
 nf_organizeAnatomy(sublist)
root_dir = '/mnt/scratch/PREK_Analysis/';

%% Preprocess data
for si = 1:length(sublist)
    sub_dir = strcat(root_dir,sublist{si});
    % Get the visit dates
    visit_dates = HCP_autoDir(sub_dir);
    % For each visit (vi = visit index)
    for vi = 1:length(visit_dates)
        % Check to see if the vist date folder is actually a date
        a = visit_dates{vi};
        sizeA = size(a);
        sizeA = sizeA(2);
        if sizeA == 8
            data_dir = strcat(root_dir,sublist{si},'/',visit_dates{vi},'/fmri');
            if exist(data_dir,'dir') == 7
                nf_preprocessfmri(data_dir)
                cd(data_dir)
                % Get functional filenames
                EPIs = dir('run*.nii');
                
                % Check how many functionals there are
                nruns = size(EPIs);
                nruns = nruns(1);
                
                EPI_names = {};
                % Create array of functional names
                for ii = 1:nruns
                    EPI_names = [EPI_names EPIs(ii).name];
                end
                
                %convert functionals, and set TR to 2.
                convert_command = 'mri_convert --out_orientation RAS';
                
                for ii = 1:nruns
                    % Convert to RAS
                    convert = [convert_command ' ' EPI_names{ii} ' ' EPI_names{ii}];
                    system(convert);
                    
                    %Set TR to 2.2
                    h = readFileNifti(EPI_names{ii});
                    h.pixdim(4) = 2;
                    h.time_units = 'sec';
                    h.data = uint16(h.data);
                    writeFileNifti(h);
                end
            end
        end
    end
end




% %% initialize vista for each subject
% for si = 4:length(sublist)
%     sub_dir = strcat(root_dir,sublist{si});
%     % Get the visit dates
%     visit_dates = HCP_autoDir(sub_dir);
%     % For each visit (vi = visit index)
%     for vi = 1:length(visit_dates)
%         % Check to see if the vist date folder is actually a date
%         a = visit_dates{vi};
%         sizeA = size(a);
%         sizeA = sizeA(2);
%         if sizeA == 8
%             data_dir = strcat(root_dir,sublist{si},'/',visit_dates{vi},'/fmri');
%             if exist(data_dir,'dir') == 7
%                 cd(data_dir)
%                 nf_initialize_vista(sublist{si},data_dir)
%                 nf_glmDenoise(data_dir)
%             end 
%         end 
%     end 
% end
