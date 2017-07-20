% Searches for the file of interest produced from long_stats_slopes and
% consolidates all of the data from this file across all subjects 
%
% Inputs:
% sub_dir: path to the directory containing all subject files
% subGroup: an array of the subject IDs for some subgroup of all of the
% subjects
% FOI: the name of the file, found in <subjectID>_template/stats/, to be
% consolidated across all subjects listed in subGroup
% meas_label: a string that is the measurement described in the FOI (should
% not contain spaces as they are the delimiter)
% outpath: path to the directory where the consolidated file will be output

function consolidate_long_stats(sub_dir,subGroup,FOI,meas_label,outpath)
% open and make file
out_file = fullfile(outpath,strcat('consol_',FOI));
system(strcat('cd',[' ',outpath]));
system(strcat('touch',[' ',out_file]));
out_id = fopen(out_file,'w');
% print first line of output: the measure label followed by all of the ROIs
fprintf(out_id,'%s ',meas_label);
for ii = 1:numel(subGroup)
    fullpath = fullfile(sub_dir,strcat(subGroup{ii},'_template'),'stats',FOI);
    if exist(fullpath,'file')
        [data,vars,cases] = tblread(fullpath,' ');
        for jj = 1:rows(vars)
            if jj == rows(vars)
                fprintf(out_id,'%s\n',strtrim(vars(jj,:)));
            else
                fprintf(out_id,'%s ',strtrim(vars(jj,:)));
            end
        end
        break
    end
end
% print out the data from all the files to the single consolidated file
for ii = 1:numel(subGroup)
    fullpath = fullfile(sub_dir,strcat(subGroup{ii},'_template'),'stats',FOI);
    if exist(fullpath,'file')
        [data,vars,cases] = tblread(fullpath,' ');
        fprintf(out_id,'%s ',cases(1,:));
        for jj = 1:cols(data)
            if jj == cols(data)
                fprintf(out_id,'%.4f\n',data(jj));
            else
                fprintf(out_id,'%.4f ',data(jj));
            end
        end
    end
end
fclose(out_id);
end