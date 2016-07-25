function [stats, test_names, subs, time_course] = runLong()
% function: runs the battery of longitudinal functions in the lme sequence
% Inputs: 
% data
% subs
% test_names
% 
% Outputs:
% 
% [stats, test_names, subs, time_course] = runLong();

%% Initialize Variables
data = [];
stats = struct;
%% Group Selection
% RAVEO
% subs = {'124_AT', '138_LA', '141_GC', '143_CH'};
% LMB
subs = {'102_RS', '110_HH', '145_AC', '150_MG', '151_RD', '152_TC', ...
        '160_EK', '161_AK', '162_EF', '163_LF', '164_SF', '170_GM', '172_TH', '174_HS', ...
        '179_GM', '180_ZD', '201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM', ...
        '207_AH', '208_LH', '210_SB', '211_LB'};
%% Test Selection
% test_names = {'WJ_BRS'};
test_names = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_BRS', 'WJ_RF', ...
                'TWRE_SWE_SS', 'TWRE_PDE_SS', 'TWRE_INDEX'};
test_names = strrep(test_names, '_', '\_');
%% Time Selection
% hours = 1; days = 2; session = 3;
time_course = 3;

%% Plotting on/off
% plotting = 0 for off; 1 for on
plotting = 0;

%% Dummy coding on/off
% dummyon = 0 for off; 1 for on
dummyon = 0;




%% Gather data and perform statistics
for ii = 1:length(test_names)
   
    test_name = test_names(ii);
        
    [sid, long_var, score, test_name] = prepLongitudinaldata(data, subs, test_name, time_course);
    if dummyon == 0
        [lme_linear, lme_quad, data_table] = lmeLongitudinaldata(sid, long_var, score);
        stats(ii).lme_quad = lme_quad;
    elseif dummyon == 1
        [lme_linear, data_table] = lmeCat(sid, long_var, score);
    end
    stats(ii).test_name = test_name;
    stats(ii).sessions = long_var;
    stats(ii).lme_linear = lme_linear;
    stats(ii).data_table = data_table;  
end


%% Plot data & Lines of best fit
if plotting == 1
    [stats] = lmeLongitudinalplot(stats, test_names, subs, time_course);
    
    % Plot histogram of growth estimates with error bars
    [stats] = lmeGrowthplot(stats, test_names, subs, time_course);
    
    % % Gather by Session data and run one-way ANOVA
    % [stats] = bySession(stats);
    
    % % plot Correlation matrix
    % [stats] = plotCorr(stats, data, subs);
elseif plotting == 0
    return;
end
return