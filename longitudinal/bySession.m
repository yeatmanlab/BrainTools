function [stats] = bySession(stats)
% Creates a dummy variable-based breakdown of the influence of each 
% session to describe differing rates of change

% %% Initialize Variables
% data = [];
% stats = struct;
% glm = struct;
% %% Group Selection
% subs = {'102_RS', '110_HH', '145_AC', '150_MG', '151_RD', '152_TC', ...
%         '160_EK', '161_AK', '162_EF', '163_LF', '164_SF', '170_GM', '172_TH', '174_HS', ...
%         '179_GM', '180_ZD', '201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM', ...
%         '207_AH', '208_LH', '210_SB', '211_LB'};
% %% Test Selection
% test_names = {'WJ_LWID_SS', 'WJ_WA_SS', 'WJ_BRS', 'WJ_RF', ...
%                 'TWRE_SWE_SS', 'TWRE_PDE_SS', 'TWRE_INDEX'};
% test_names = strrep(test_names, '_', '\_');
% %% Time Selection
% % hours = 1; days = 2; session = 3;
% time_course = 3;
% 
% %% Gather data and perform statistics
% for ii = 1:length(test_names)
%    
%     test_name = test_names(ii);
%         
%     [sid, long_var, score, test_name] = prepLongitudinaldata(data, subs, test_name, time_course);
%     [lme_linear, lme_quad, data_table] = lmeLongitudinaldata(sid, long_var, score);
%    
%     stats(ii).test_name = test_name;
%     stats(ii).sessions = long_var;
%     stats(ii).lme_linear = lme_linear;
%     stats(ii).lme_quad = lme_quad;
% %     stats(ii).logistic = logistic_stats;
%     stats(ii).data_table = data_table;  
% end
% 

%% 
for ii = 1:length(stats)
    % Pull information from data struct
    test_name = char(stats(ii).test_name);
    test_name = strrep(test_name, '\_', '_');
    subs = stats(ii).data_table.sid;
    sessions = stats(ii).sessions;
    score = stats(ii).data_table.score;
    % Create categories for sessions
    sessions = categorical(sessions);
    % Create dummy variable matrix 
    dummy = dummyvar(sessions);
    
%     [mean, std, sem, var] = grpstats(score, ...
%                 sessions, {'mean', 'std', 'sem', 'var'});   
%     figure; hold;
%     bar(mean, 'EdgeColor', 'k', 'FaceColor', 'b');
%     ax = gca;
%     ax.XAxis.Limits = [0 7];
%     ax.YAxis.Limits = [50 100];
%     errorbar(ax, mean, sem, '-k');
%     xlabel('Session');
%     ylabel('Mean Score');
%     title([test_name, ' by Session']);
    
%      % Save image
%     fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.png','bySession', test_name, date);
%     print(fname, '-dpng'); 
    
    
   
    [A, tbl, s] = anova1(score, sessions, 'off');
    [c, m, h] = multcompare(s);
    c;
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.fig','MultCompare', test_name, date);
    savefig(h, fname); 
    
    
    figure; hold; 
    boxplot(score, sessions, 'Notch', 'on');
    title(['ANOVA1 for ', test_name]);
    xlabel('Session');
    ylabel('One-Way Analysis of Variance');
    t = text(4, 50, num2str(A));
    t.FontSize = 12;
    
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.png','ANOVA', test_name, date);
    print(fname, '-dpng'); 
    
    
    stats(ii).anovap = A; 
    stats(ii).anovatble = tbl; 
    stats(ii).anovastats = s;
    stats(ii).multcompare.c = c;
    stats(ii).multcompare.m = m;
    stats(ii).multcompare.h = h;
    
end







end