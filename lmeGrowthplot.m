function [stats] = lmeGrowthplot(stats, test_names, subs, time_course)
% [stats] = lmeGrowthplot(stats, test_names, subs, time_course)
% 
% Function: plots histogram of growth statistics of lme data
% Inputs: stats, type struct; test_names; subs; time_course
% Outputs: figure, type histogram
% Example:

%% Linear
figure; hold;
for ii = 1:length(test_names)
    linear_data(ii,:) = table(test_names(ii), stats(ii).lme_linear.Coefficients.Estimate(2), stats(ii).lme_linear.Coefficients.SE(2));
    linear_data.Properties.VariableNames = {'test_name', 'Growth', 'SE'};    
end

h = bar(linear_data.Growth, 'FaceColor', 'w', 'EdgeColor', 'k');
errorbar(linear_data.Growth, linear_data.SE, 'kx');

for jj = 1:length(test_names)
   text(jj, linear_data.Growth(jj) + linear_data.SE(jj) + 0.01, ...
        num2str(round(stats(jj).lme_linear.Coefficients.pValue(2), 3)), ...
        'HorizontalAlignment', 'center', 'Color', 'b');
end
   

% Format
ylabel('Growth Estimate'); xlabel('Test Name');
ax = gca;
ax.XTick = 1:length(test_names);
ax.XTickLabel = test_names;
ax.XTickLabelRotation = 45;
title('Linear Growth Estimate by Test');

% Save image
    fname = sprintf('~/Desktop/figures/LMB/linearGrowthEst-%s.png', date);
    print(fname, '-dpng'); 

%% Quadratic
% Determine if quadratic fit is necessary
if time_course ~= 3
    
    figure; hold;
    
    for ii = 1:length(test_names)
        quad_data(ii,:) = table(test_names(ii), stats(ii).lme_quad.Coefficients.Estimate(3), stats(ii).lme_quad.Coefficients.SE(3));
        quad_data.Properties.VariableNames = {'test_name', 'Growth', 'SE'};
    end
    h = bar(quad_data.Growth, 'FaceColor', 'g', 'EdgeColor', 'k');
    errorbar(quad_data.Growth, linear_data.SE, 'k');
    
    for jj = 1:length(test_names)
        text(jj, quad_data.Growth(jj) + quad_data.SE(jj) + 0.01, ...
            num2str(round(stats(jj).lme_quad.Coefficients.pValue(3), 3)), ...
            'HorizontalAlignment', 'center', 'Color', 'b');
    end
        
    % Format
    ylabel('Growth Estimate'); xlabel('Test Name');
    ax = gca;
    ax.XTick = 1:length(test_names);
    ax.XTickLabel = test_names;
    ax.XTickLabelRotation = 45;
    title('Quadratic Growth Estimate by Test');
        
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/quadGrowthEst-%s.png', date);
    print(fname, '-dpng');
    
else
    return;
end
