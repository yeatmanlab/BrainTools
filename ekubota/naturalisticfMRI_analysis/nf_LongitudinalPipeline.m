full_sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184', 'PREK_1798','PREK_1302','PREK_1460','PREK_1110','PREK_1756',...
    'PREK_1966','PREK_1750','PREK_1940','PREK_1262','PREK_1113'};

sublist = {'PREK_1762','PREK_1964','PREK_1887','PREK_1673','PREK_1869','PREK_1676'};
sublist = {'PREK_1112','PREK_1691','PREK_1916','PREK_1951','PREK_1901'};
sublist = {'PREK_1715','PREK_1921','PREK_1208','PREK_1271','PREK_1372',...
    'PREK_1939','PREK_1868','PREK_1505'};

nf_organizeLongitudinalData(sublist)
% nf_organizeAnatomy(sublist)
root_dir = '/mnt/scratch/PREK_Analysis/';

%% Preprocess data
for si = 1:length(sublist)
    data_dir = strcat(root_dir,sublist{si},'/longitudinal');
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

%% next steps after freesurfer is run.
nf_copyParfilesLongitudinal(sublist)
nf_glmDenoiseSubsLongitudinal(sublist);
%nf_organizeAnatomy(sublist);
%nf_saveMeshsubs(sublist);
nf_alignFunctionaltoVolume(sublist,session);
% nf_writeOutfsROIs(sublist)
% nf_divideFSRois(sublist)
% nf_writeOutContrastAsNifti(sublist);
% nf_reliabilityCorrAsNifti(sublist);
[C,include,exclude] = nf_excludeMotion(sublist,session);
% nf_MapFmriToFS(include);
% nf_fsStats;

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
