function [stats] = lmeEstimateplot(stats, test_names, subs, time_course);
% [stats] = lmeEstimateplot(stats, test_names, subs);
%
% Function:

% Deside on single or individual plots
% if decision == 1; single. if decision == 2; individual
decision = 1;

if decision == 1
    %% Create Single LME Mean Model Plot
    figure; hold;
    dmap = lines;
    dmap = vertcat(dmap(2,:), dmap(1,:), dmap(5,:));
    num_sessions = 5; % number of sessions including session 0
    sessions = [0 1 2 3 4];
    %Plot Individual Trends
    for ii = 1:length(test_names)
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
        h = plot(sessions', estimates, '-o', 'Color', dmap(ii,:), 'MarkerFaceColor', dmap(ii,:), 'LineWidth', 2, 'MarkerSize', 6, 'MarkerEdgeColor', dmap(ii,:));
%         
    end
    
    %Format Plot
    ax = gca;
    ax.XLim = [-0.1 4.1];
    ax.YLim = [70 100];
    ax.XAxis.TickValues = [0 1 2 3 4];
    xlabel('Session'); ylabel('Standard Score');
    title('Growth in Reading Skill');
    grid('on');
%     axis('tight');
    legend(strrep(test_names, '_', '\_'), 'Location', 'eastoutside');

    
    %Add Error Bars
    for ii = 1:length(test_names)
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
        errorbar(sessions', estimates, se, '.k', 'Color', dmap(ii,:), 'LineWidth', 2);
    end
    % Save image
    test = num2str(cell2mat(test_names(ii)));
    test = strrep(test, '\_', '-');
    fname = sprintf('C:/Users/Patrick/Desktop/figures/LMB/%s-%s-%s.eps', 'LMEestimate', 'Composites', date);
    print(fname, '-depsc');
    
elseif decision == 2
    %% Plot Individual Growth Plots
    for ii = 1:length(test_names)
        num_sessions = 5; % number of sessions including session 0
        sessions = [0 1 2 3 4];
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
        
        % set figure
        figure; hold;
        % Plot curves
        h = bar(sessions', estimates, 'w');
        % Plot error bars
        errorbar(sessions', estimates, se, '.k');
        
        % Add */** based on significance of pValues
        for num = 1:num_sessions
            if p(num) <= 0.001
                text(sessions(num), estimates(num) + se(num) + 2, ...
                    '**', 'HorizontalAlignment', 'center', 'Color', 'b');
            elseif p(num) <= 0.05
                text(sessions(num), estimates(num) + se(num) + 2, ...
                    '*', 'HorizontalAlignment', 'center', 'Color', 'b');
            end
        end
        
        %Format Plot
        ax = gca;
        ax.XLim = [-0.5000 4.5000];
        ax.YLim = [70, (max(estimates) + 5)];
        ax.XAxis.TickValues = [0 1 2 3 4];
        xlabel('Session'); ylabel('Standard Score');
        title([strrep(test_names(ii), '_', '\_'), ' LME Mean Growth']);
        grid('on');
        axis('tight');
        legend(h, test_names, 'Location', 'eastoutside');
        
    end
    
%     % Save image
%     test = num2str(cell2mat(test_names(ii)));
%     test = strrep(test, '\_', '-');
%     fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.eps', 'LMEestimate', test, date);
%     print(fname, '-depsc');
end
end


