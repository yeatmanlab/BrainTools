function [stats] = lmeEstimateplot(stats, test_names, subs, time_course);
% [stats] = lmeEstimateplot(stats, test_names, subs);
% 
% Function: 


%% Time Variable
if time_course == 1
    x_name = 'hours';
    xx = [0 50 100 150 200];
elseif time_course == 2
    x_name = 'days'; 
    xx = [0 10 20 30 40 50 60 70]; 
elseif time_course == 3
    x_name = 'session';
    xx = [1 2 3 4]; 
    
end
% if doing a single plot
figure; hold
c = jet(length(test_names));

%% Create plots
for ii = 1:length(test_names)
    num_sessions = 4; % number of sessions including session 0
    sessions = [1 2 3 4]; 
    estimates = zeros(num_sessions, 1);
    se = zeros(num_sessions, 1);
    p = zeros(num_sessions, 1);
    for num = 1:num_sessions
        estimates(num, 1) = stats(ii).lme_linear.Coefficients.Estimate(num);
        se(num, 1) = stats(ii).lme_linear.Coefficients.SE(num);
        p(num, 1) = round(stats(ii).lme_linear.Coefficients.pValue(num), 3); 
    end
    
    for num = 2:num_sessions
       estimates(num, 1) = (estimates(1,1) + estimates(num, 1)); 
    end
%     % if doing individual plots
%     figure; hold;
    plot(sessions', estimates);
    errorbar(sessions', estimates, se, '.k');
%     for num = 1:num_sessions
%         if p(num) <= 0.001
%             text(sessions(num), estimates(num) + se(num) + 2, ...
%                 '**', 'HorizontalAlignment', 'center', 'Color', 'b');
%         elseif p(num) <= 0.05
%             text(sessions(num), estimates(num) + se(num) + 2, ...
%                 '*', 'HorizontalAlignment', 'center', 'Color', 'b');
%         end
%     end

    ax = gca;   
    ax.XLim = [0.5000 4.5000];
    ax.XAxis.TickValues = [1 2 3 4];
%     ax.YLim = [70 (max(estimates) + 5)];

    xlabel('Session'); ylabel('LME Estimate');
    title([strrep(test_names(ii), '_', '\_'), 'LME Estimate']);
    grid('on');
    axis('tight');
     
%     % Save image
%     test = num2str(cell2mat(test_names(ii)));
%     test = strrep(test, '\_', '-');
%     fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.png', 'LMEestimate', test, date);
%     print(fname, '-dpng');
end
% if single plot
   l = legend(test_names, 'Location', 'eastoutside');
    ax.YLim = [70 110];


return