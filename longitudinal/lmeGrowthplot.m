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
    if stats(jj).lme_linear.Coefficients.pValue(2) <= 0.001
        text(jj, linear_data.Growth(jj) + linear_data.SE(jj) + .002, ...
            '**', 'HorizontalAlignment', 'center', 'Color', 'b');
    elseif stats(jj).lme_linear.Coefficients.pValue(2) <= 0.05
        text(jj,linear_data.Growth(jj) + linear_data.SE(jj) + .002, ...
            '*', 'HorizontalAlignment', 'center', 'Color', 'b');
    end
end

% Format
ylabel('Growth Estimate'); xlabel('Test Name');
ax = gca;
ax.XTick = 1:length(test_names);
ax.XTickLabel = strrep(test_names, '_', '\_');
ax.XTickLabelRotation = 45;
title('Linear Growth Estimate by Test');

% Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-linearGrowthEst-%s.eps', 'hours', date);
    print(fname, '-depsc'); 

%% Quadratic
% Determine if quadratic fit is necessary
if time_course ~= 3
    
    figure; hold;
    
    for ii = 1:length(test_names)
        quad_data(ii,:) = table(test_names(ii), stats(ii).lme_quad.Coefficients.Estimate(3), stats(ii).lme_quad.Coefficients.SE(3));
        quad_data.Properties.VariableNames = {'test_name', 'Growth', 'SE'};
    end
    h = bar(quad_data.Growth, 'FaceColor', 'w', 'EdgeColor', 'k');
    errorbar(quad_data.Growth, linear_data.SE, 'kx');
    
    for jj = 1:length(test_names)
        if stats(jj).lme_quad.Coefficients.pValue(3) <= 0.001
            text(jj, estimates(num) + se(num) + 2, ...
                '**', 'HorizontalAlignment', 'center', 'Color', 'b');
        elseif stats(jj).lme_quad.Coefficients.pValue(3) <= 0.05
            text(jj, estimates(num) + se(num) + 2, ...
                '*', 'HorizontalAlignment', 'center', 'Color', 'b');
        end
    end
        
    % Format
    ylabel('Growth Estimate'); xlabel('Test Name');
    ax = gca;
    ax.XTick = 1:length(test_names);
    ax.XTickLabel = strrep(test_names, '_', '\_');
    ax.XTickLabelRotation = 45;
    title('Quadratic Growth Estimate by Test');
        
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-quadGrowthEst-%s.eps', 'hours', date);
    print(fname, '-depsc');
    
else
    return;
end
