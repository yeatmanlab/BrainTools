% This is a script that reads in behavioral data and organizes it 
% for the purposes of performing statistics and plotting techniques

% Inputs: data (in the form of a .xlsx file) & subs (the subjects that you
%         you would like to use) & test_names (the tests of interest)
% Outputs: vectorized data for each variable in the data set
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
%% Test Selection; subgroups
all = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_OR_SS', 'WJ_SRF_SS', 'WJ_MFF_SS', 'WJ_CALC_SS', 'WJ_BRS', 'WJ_RF',...
                'TWRE_SWE_SS', 'TWRE_PDE_SS', 'TWRE_INDEX', 'WASI_FS2', 'CTOPP_ELISION_SS', 'CTOPP_BW_SS', 'CTOPP_PI_SS',...
                'CTOPP_RAPID', 'CTOPP_PA'};
wj  = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_OR_SS', 'WJ_SRF_SS', 'WJ_MFF_SS', 'WJ_CALC_SS', 'WJ_BRS', 'WJ_RF'};
twre= {'TWRE_SWE_SS', 'TWRE_PDE_SS', 'TWRE_INDEX'};
wasi= {'WASI_FS2'};
ctopp= {'CTOPP_ELISION_SS', 'CTOPP_BW_SS', 'CTOPP_PI_SS','CTOPP_RAPID', 'CTOPP_PA'};
math = {'WJ_MFF_SS', 'WJ_CALC_SS'};
basic = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_BRS', 'WJ_RF', 'TWRE_SWE_SS', 'TWRE_PDE_SS', 'TWRE_INDEX'};
%% Selections
% test group options: all, wj, twre, wasi, ctopp, math, and basic
test_names = basic; 
% time course options: (1) hours, (2) days, (3) sessions
time_course = 2; 
% enter sessions of interest, if applicable
usesessions = []; 
% dummy variable options: (0) off, (1) on
dummyon = 0;
% longitudinal plot options: (0) off, (1) on
long_plot = 0;
% growth plot options: (0) off, (1) on
growth_plot = 0;
% lme estimate plot options: (0) off, (1) on
lmestimate_plot = 0;

%% Data set
%     If using a Mac/Linux
      [~, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
%     If using a PC
%     [~, ~, data] = xlsread('C:\Users\Patrick\Desktop/NLR_Scores.xlsx');
%% Group Selection
% RAVE-O Pilot Intervention Group
% subs = {'124_AT', '138_LA', '141_GC', '143_CH'};
% Lindamood-Bell Subjects
subs = {'102_RS', '110_HH', '145_AC', '150_MG', '151_RD', '152_TC', ...
        '160_EK', '161_AK', '162_EF', '163_LF', '164_SF', '170_GM', ...
        '172_TH', '174_HS', '179_GM', '180_ZD', '201_GS', '202_DD', ...
        '203_AM', '204_AM', '205_AC', '206_LM', '207_AH', '208_LH', ...
        '210_SB', '211_LB'};


%% Gather data
stats = struct; % initialize the struct to store all data per test for analysis
for ii = 1:length(test_names);    
    % run readData function to gather data of interest
    [sid, long_var, score, test_name] = readData(data, subs, test_names(ii), time_course, usesessions);
    % gather lme statistics using lmeCalc function
    [lme_linear, lme_quad, data_table] = lmeCalc(sid, long_var, score, dummyon);  
    % Collate data into stats struct
    stats(ii).test_name = test_name; 
    stats(ii).lme_linear = lme_linear;
    stats(ii).data_table = data_table;  
    stats(ii).lme_quad = lme_quad;
end

%% Plotting Techniques
if long_plot == 1
   [stats] = lmeLongplot(stats, test_names, subs, time_course); 
end
    
if growth_plot == 1
   [stats] = lmeGrowthplot(stats, test_names, subs, time_course);
end

if lmestimate_plot == 1
   [stats] = lmeEstimateplot(stats, test_names, subs, time_course);
end
