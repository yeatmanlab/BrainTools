%% lmeLongitudinaldata

% Select group of Subjects
% read data from Desktop
[tmp, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
% gather column headings
data_ref = data(1,:);
% remove data headers from data
data = data(2:end,:);
% create array of subjects of interest
subs = {'201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM'};
% find all rows for subjects of interest
data_indx_tmp = [];
data_indx     = [];
for ii = 1:numel(subs)
    data_indx_tmp = find(strcmp(data(:, strcmp(data_ref, 'Subject')), subs(ii)));
    data_indx = vertcat(data_indx, data_indx_tmp);
end
% create refined data array for data of interest
sid = {}; sessnum = {}; time = {}; hours = {}; wj_brs = {};
for ii = 1:numel(data_indx)
    sid         = vertcat(sid, data(data_indx(ii), strcmp(data_ref, 'Subject')));
    sessnum     = vertcat(sessnum, data(data_indx(ii), strcmp(data_ref, 'Visit')));
    time        = vertcat(time, data(data_indx(ii), strcmp(data_ref, 'Time')));
    hours       = vertcat(hours, data(data_indx(ii), strcmp(data_ref, 'Hours')));
    % READING TESTS
    wj_brs      = vertcat(wj_brs, data(data_indx(ii), strcmp(data_ref, 'WJ_BRS')));
%     wj_rf(ii)       = data(data_indx(ii), strcmp(data_ref, 'WJ_RF'));
%     wj_lwid(ii)     = data(data_indx(ii), strcmp(data_ref, 'WJ_LWID_SS'));
%     wj_wa(ii)       = data(data_indx(ii), strcmp(data_ref, 'WJ_WA_SS'));
%     twre_swe(ii)    = data(data_indx(ii), strcmp(data_ref, 'TOWRE_SWE_SS'));
%     twre_pde(ii)    = data(data_indx(ii), strcmp(data_ref, 'TOWRE_PDE_SS'));
%     twre_indx(ii)   = data(data_indx(ii), strcmp(data_ref, 'TWRE_INDEX'));
%     wasi(ii)        = data(data_indx(ii), strcmp(data_ref, 'WASI_FS2'));
%     elision(ii)     = data(data_indx(ii), strcmp(data_ref, 'CTOPP_ELISION_SS'));
%     ctopp_pa(ii)    = data(data_indx(ii), strcmp(data_ref, 'CTOPP_PA'));
%     ctopp_rn(ii)    = data(data_indx(ii), strcmp(data_ref, 'CTOPP_RAPID'));
%     % WORD LISTS
%     wl_4let(ii)     = data(data_indx(ii), strcmp(data_ref, 'WL_4let'));
%     wl_5let(ii)     = data(data_indx(ii), strcmp(data_ref, 'WL_5let'));
end

% Convert cell arrays to tables
% sid         = cell2mat(sid)';
% sessnum     = cell2mat(sessnum)';
% time        = cell2mat(time)';
% hours       = cell2mat(hours)';



%% Name Reading Score of Interest

% reading_score = cell2mat(wj_brs)';
reading_score = wj_brs;

%% Create Variations for Model Testing
for ii = 1:numel(data_indx)
   hours_sq(ii) = hours(ii)^2; 
end

%% Create DataSet
data_table = table(sid, hours, reading_score);


% Make sid a categorical variable
data_table.sid = nominal(data_table.sid);
% Fit the model where we predict BR as changing linearly with the number of
% hours of intervention
lme = fitlme(data_table, 'reading_score ~ hours + (1|sid)');
% Fit the model where we predict BR as changing quadratically with hours of
% intervention
lme2 = fitlme(data_table, 'reading_score ~ hours + hours2 + (1|sIds)');






