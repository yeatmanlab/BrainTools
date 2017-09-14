function afq = HCP_bxPrep(fullBxFile, afq, afq_outname)
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
% fullBxFile = '/mnt/scratch/Behavioral/unrestricted_user_3_29_2016_12_54_53.csv'
% afq = HCP_behPrep(fullBxFile, afq)

% Import behavioral data as a table
T = readtable(fullBxFile);

% Create vector to index subject rows
subs = zeros(1,numel(afq.sub_names));

% Find subjects of interest LET'S WORRY ABOUT THIS CODE AT SOME POINT
for ii = 1:numel(afq.sub_names)
    subs(ii) = find(T.Subject == str2double(afq.sub_names{ii}));
    subsAFQ(ii) = str2double(afq.sub_names{ii});
end

% Construct new table with only data from subjects of interest
% (This is an unnecessary intermediary step that will be removed after
% testing)
subT = T(subs,:);

% Check that the rows match the afq structure
if ~all(subsAFQ' == subT.Subject)
    error('\n row mismatch between table and afq struct\n')
end

% Get the names of all the variables in the table
vn = fieldnames(subT);

% Write all info as metadata. Maintains categories as assigned by HCP. Call
% afq.metadata to see all function names.
for ii = 1:length(vn)
    afq = AFQ_set(afq, 'metadata', vn{ii}, subT.(vn{ii}));
end

if exist('afq_outname', 'var') && ~isempty(afq_outname)
   save(afq_outname, 'afq') 
end
