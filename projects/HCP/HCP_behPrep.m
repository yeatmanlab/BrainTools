function afq = HCP_behPrep(fullBxFile, afq)
% This function automatically extracts the HCP behavioral data for subjects
% being run in the current trial. Current default is to extract all
% data associated with each subject, but this function will likely be pared
% down to focus on data of interest to present inquiry....
%
% fullBxFile - points to behavioral data *.csv file downloaded from
% dbConnectome
% 
% afq - uses output from AFQ_run_sge; indexes behavioral metadata back into
% afq and writes file back out
%
%
% example:
% dirList = HCP_autoDir(baseDir)
% fullBxFile = '/mnt/scratch/Behavioral/unrestricted_user_3_29_2016_12_54_53.csv'
% behFile = HCP_behPrep(fullBxFile, dirlist)

% Import behavioral data as a table
T = readtable(fullBxFile);

% Create vector to index subject rows
subs = zeros(1,numel(afq.sub_names));

% Find subjects of interest
for ii = 1:numel(afq.sub_names)
    subs(ii) = find(T.Subject == str2double(afq.sub_names{ii}));
end

% Construct new table with only data from subjects of interest
% (This is an unnecessary intermediary step that will be removed after
% testing)
subT = T(subs,:);

% Write all info as metadata
afq.metadata = table2struct(subT);
    