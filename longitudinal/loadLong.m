% This is a script that reads in behavioral data and organizes it 
% for the purposes of performing statistics and plotting techniques

% Inputs: data (in the form of a .xlsx file) & subs (the subjects that you
%         you would like to use) & test_names (the tests of interest)
% Outputs: vectorized data for each variable in the data set
%% Set up
clear all;
clc; 
%% Data set
data = [];
%% Group Selection
% RAVE-O Pilot Intervention Group
% subs = {'124_AT', '138_LA', '141_GC', '143_CH'};
% Lindamood-Bell Subjects
subs = {'102_RS', '110_HH', '145_AC', '150_MG', '151_RD', '152_TC', ...
        '160_EK', '161_AK', '162_EF', '170_GM', '172_TH', '174_HS', ...
        '179_GM', '180_ZD', '201_GS', '202_DD', '203_AM', '204_AM', ...
        '205_AC', '206_LM', '207_AH', '208_LH', '210_SB', '211_LB'};
% Subjects removed from set
%     '163_LF', '164_SF' % have not yet gotten a second session
%% Test Selection
% test_names = {'WJ_MFF_SS', 'WJ_CALC_SS'};
test_names = {'LWID', 'WA', 'OR', 'SRF', 'MFF', 'CALC', 'WJ_BRS', 'WJ_RF',...
                'SWE', 'PDE', 'TWRE_INDEX', 'WASI', 'ELISION', 'BW', 'PI',...
                'CTOPP_RAPID', 'CTOPP_PA'};
%% Argument Checking
if ~exist('data', 'var') || isempty(data)
%     If using a Mac
      [~, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
%     If using a PC
%     [~, ~, data] = xlsread('C:\Users\Patrick\Desktop/NLR_Scores.xlsx');
end

if ~exist('subs', 'var') || isempty(subs)
   error('Please enter the subjects you would like to use');  
   return
end

if ~exist('test_names', 'var') || isempty(test_names)
   error('Please enter the reading test of interest');
   return
end
%% Format data
% gather column headings
data_ref = data(1,:);
% add '\' preceding each "_" for nicer looking titles/formatting
% data_ref = strrep(data_ref, '_', '\_');
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
% initialize empty arrays for time variables
sid = []; sessions = []; days = []; hours = [];
% vertcat each time course variable
for subj = 1:numel(data_indx)
    sid        = vertcat(sid, data(data_indx(subj), strcmp(data_ref, 'Subject')));
    sessions    = vertcat(sessions, data(data_indx(subj), strcmp(data_ref, 'LMB_session')));
    days       = vertcat(days, data(data_indx(subj), strcmp(data_ref, 'Time')));
    hours      = vertcat(hours, data(data_indx(subj), strcmp(data_ref, 'Hours')));   
end

% Convert cell arrays to variables suitable for use with dataset()
hours = cell2mat(hours); 
sessions = cell2mat(sessions);
days  = cell2mat(days); 

%% Gather reading score/s of interest
% Intialize variables
tests = struct; % struct where scores will be stored
tests.names = test_names; % place test names in struct
% Initilize empty array for each score
tests.lwid = []; tests.wa = []; tests.or = []; tests.srf = []; 
tests.mff = []; tests.calc = []; tests.wj_brs = []; tests.wj_rf = [];
tests.swe = []; tests.pde = []; tests.twre_index = []; tests.wasi = []; 
tests.elision = []; tests.bw = []; tests.pi = []; 
tests.ctopp_rapid = []; tests.ctopp_pa = [];   
% vertcat each reading score variable and store in struct
for subj = 1:numel(data_indx)
    tests.lwid         = vertcat(tests.lwid, data(data_indx(subj), strcmp(data_ref, 'WJ_LWID_SS')));
    tests.wa           = vertcat(tests.wa, data(data_indx(subj), strcmp(data_ref, 'WJ_WA_SS')));
    tests.or           = vertcat(tests.or, data(data_indx(subj), strcmp(data_ref, 'WJ_OR_SS')));
    tests.srf          = vertcat(tests.srf, data(data_indx(subj), strcmp(data_ref, 'WJ_SRF_SS')));
    tests.mff          = vertcat(tests.mff, data(data_indx(subj), strcmp(data_ref, 'WJ_MFF_SS')));
    tests.calc         = vertcat(tests.calc, data(data_indx(subj), strcmp(data_ref, 'WJ_CALC_SS')));
    tests.wj_brs       = vertcat(tests.wj_brs, data(data_indx(subj), strcmp(data_ref, 'WJ_BRS')));
    tests.wj_rf        = vertcat(tests.wj_rf, data(data_indx(subj), strcmp(data_ref, 'WJ_RF')));
    tests.swe          = vertcat(tests.swe, data(data_indx(subj), strcmp(data_ref, 'TWRE_SWE_SS')));
    tests.pde          = vertcat(tests.pde, data(data_indx(subj), strcmp(data_ref, 'TWRE_PDE_SS')));
    tests.twre_index   = vertcat(tests.twre_index, data(data_indx(subj), strcmp(data_ref, 'TWRE_INDEX')));
    tests.wasi         = vertcat(tests.wasi, data(data_indx(subj), strcmp(data_ref, 'WASI_FS2')));
    tests.elision      = vertcat(tests.elision, data(data_indx(subj), strcmp(data_ref, 'CTOPP_ELISION_SS')));
    tests.bw           = vertcat(tests.bw, data(data_indx(subj), strcmp(data_ref, 'CTOPP_BW_SS')));
    tests.pi           = vertcat(tests.pi, data(data_indx(subj), strcmp(data_ref, 'CTOPP_PI_SS')));
    tests.ctopp_rapid  = vertcat(tests.ctopp_rapid, data(data_indx(subj), strcmp(data_ref, 'CTOPP_RAPID')));
    tests.ctopp_pa     = vertcat(tests.ctopp_pa, data(data_indx(subj), strcmp(data_ref, 'CTOPP_PA')));
end
% Convert Scores to MATLAB variables and arrange in struct
tests.lwid = cell2mat(tests.lwid);  tests.wa = cell2mat(tests.wa);
tests.or = cell2mat(tests.or); tests.srf = cell2mat(tests.srf);
tests.mff = cell2mat(tests.mff); tests.calc = cell2mat(tests.calc);
tests.wj_brs = cell2mat(tests.wj_brs); tests.wj_rf = cell2mat(tests.wj_rf);
tests.swe = cell2mat(tests.swe); tests.pde = cell2mat(tests.pde);
tests.twre_index = cell2mat(tests.twre_index); tests.wasi = cell2mat(tests.wasi);
tests.elision = cell2mat(tests.elision); tests.bw = cell2mat(tests.bw);
tests.pi = cell2mat(tests.pi); tests.ctopp_rapid = cell2mat(tests.ctopp_rapid);
tests.ctopp_pa = cell2mat(tests.ctopp_pa);

