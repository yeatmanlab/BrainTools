function [sid, hours, test_name, reading_score] = prepLongitudinaldata(data, subs, test_name)
% Prepares data for lmeLongitudinaldata and plotLongitudinaldata
% 
% [lme, lme2, data_table] = lmeLongitudinaldata(data, test_name, subs)
% 
% Inputs: 
% data
% test_name
% subs
% 
% Outputs:
% 
% sid
% hours
% reading_score
% 
% Example:
% 
% data = []; subs = {'...', '...', '...'}; test_name = 'WJ_BRS'; 
% [sid, hours, test_name, reading_score] = prepLongitudinaldata(data, subs, test_name);


%% Argument Checking
if ~exist('data', 'var') || isempty(data)
    %[~, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
    [~, ~, data] = xlsread('C:\Users\Patrick\Desktop/NLR_Scores.xlsx');
end

if ~exist('subs', 'var') || isempty(subs)
   error('Please enter the subjects you would like to use');  
   return
end

if ~exist('test_name', 'var') || isempty(test_name)
   error('Please enter the reading test of interest');
   return
end


%% Select group of Subjects

% gather column headings
data_ref = data(1,:);
% remove data headers from data
data = data(2:end,:);
% find all rows for subjects of interest
data_indx_tmp = [];
data_indx     = [];
for subj = 1:numel(subs)
    data_indx_tmp = find(strcmp(data(:, strcmp(data_ref, 'Subject')), subs(subj)));
    data_indx = vertcat(data_indx, data_indx_tmp);
end
% create refined data array for data of interest
% initialize empty arrays
sid = []; sessnum = []; time = []; hours = [];

% vertcat each reading test variable
for subj = 1:numel(data_indx)
    sid        = vertcat(sid, data(data_indx(subj), strcmp(data_ref, 'Subject')));
    sessnum    = vertcat(sessnum, data(data_indx(subj), strcmp(data_ref, 'Visit')));
    time       = vertcat(time, data(data_indx(subj), strcmp(data_ref, 'Time')));
    hours      = vertcat(hours, data(data_indx(subj), strcmp(data_ref, 'Hours')));   
end

% Convert cell arrays to variables suitable for use with dataset()
hours       = cell2mat(hours);



%% Gather Reading Score of Interest
% intialize variable
reading_score = []; 

% vertcat the data into a cell matrix
for subj = 1:numel(data_indx)
reading_score = vertcat(reading_score, data(data_indx(subj), strcmp(data_ref, test_name)));
end




return


