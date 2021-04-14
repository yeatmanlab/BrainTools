% read in the data
code_dir = '/mnt/disks/scratch/PREK_Analysis/code/BrainTools/ekubota/naturalisticfMRI_analysis/';
T = readtable(fullfile(code_dir,'behavioral.csv'));

full_sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184', 'PREK_1798','PREK_1302','PREK_1460','PREK_1110','PREK_1756',...
    'PREK_1966','PREK_1750','PREK_1940','PREK_1262','PREK_1113','PREK_1241'};

[~,include,~] = nf_excludeMotion(full_sublist,'ses-pre')

subs = include(:,1);

% Visit codes: 
%   61 - letter pre
%   62 - letter post 
%   63 - letter one year 
%   64 - language pre
%   65 - language post 
%   66 - language one year 

% Let's filter and get only the pre-camp visits

T = T(T.study_name == 61 | T.study_name == 64,:);
camp = [];

for s = 1:length(subs)
    [tok,rem] = strtok(subs{s},'_');
    [sid,~] = strtok(rem,'_');
    sid = str2double(sid);
    camp_code = T(T.record_id == sid,3).study_name;
    if camp_code == 61
        camp{s} = 'letter';
    elseif camp_code == 64 
        camp{s} = 'language';
    end 
end 
    
camp = camp';
%filename = '/postcamp_labels';
%h5write('data.h5',filename,camp)
writecell(camp,fullfile(code_dir,'precamp_labels.csv'))
