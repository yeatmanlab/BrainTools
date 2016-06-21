function [stats] = runLong()
% runLong
% 
% 
% function: runs the battery of longitudinal functions in the lme sequence
% 
% Inputs:
% 
% data
% subs
% test_names
% 
% Outputs:
% 
% 
% 
% [stats] = runLong();

data = [];
stats = struct;
%% Group Selection
% RAVEO
% subs = {'124_AT', '138_LA', '141_GC', '143_CH'};
% LMB
subs = {'152_TC', '201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM'};
%% Test Selection
test_names = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_BRS', 'WJ_RF'};
test_names = strrep(test_names, '_', '\_');
%% Time Selection
% hours = 1; days = 2
time_course = 2;

%% Optional input format
% subs = input('What subjects will you be using? Enter as {..., ..., ...} ');
% test_names = input('What tests will you be using? Enter as {..., ..., ...} '); 



%% Gather data and perform statistics
for ii = 1:length(test_names)
   
    test_name = test_names(ii);
        
    [sid, hours, time, test_name, reading_score] = prepLongitudinaldata(data, subs, test_name);
    [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, time, test_name, reading_score, time_course);
   
    stats(ii).test_name = test_name;
    stats(ii).lme0 = lme0;
    stats(ii).lme = lme;
    stats(ii).lme1 = lme1;
    stats(ii).lme2 = lme2;
    stats(ii).data_table = data_table;
    
end

% Plot data & Lines of best fit
[stats] = lmeLongitudinalplot(stats, test_names, subs, time_course);

% Plot histogram of growth estimates with error bars
[stats] = lmeGrowthplot(stats, test_names, subs, time_course);



return