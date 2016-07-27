function [lme_linear, data_table] = lmeCat(sid, long_var, score)
% function to calculate linear fit using categorical variable

%% Create Variations for Model Testing

% Center each individual's reading scores
s = unique(sid);
% for ii = 1:length(s)
%    index = find(strcmp(s(ii),sid));
%    total = 0;
%    for jj = 1:length(index)
%        total = plus(total, score(index(jj))); 
%    end
%    avg = total/length(index);
%    
%    for kk = 1:length(index);
%        score_sq_unique(index(kk), 1) = score(index(kk)) - avg;
%    end
% end
% score_adj = score_sq_unique;


long_var = categorical(long_var);

%% Create DataSet
data_table = dataset(sid, long_var, score);


%% Calculate LME fit
% Make sid a categorical variable
data_table.sid = categorical(data_table.sid);
% Fit the model on the centered data as changing linearly 
lme_linear = fitlme(data_table, 'score ~ long_var + (long_var|sid)');

% % Growth Plot
% figure; hold;
% 
%     y = [stats(1).lme_linear.Coefficients.Estimate(2:5); stats(2).lme_linear.Coefficients.Estimate(2:5); stats(3).lme_linear.Coefficients.Estimate(2:5); ... 
%         stats(4).lme_linear.Coefficients.Estimate(2:5)]
%     bar(y); 
%     linear_data(ii,:) = table(test_names(ii), stats(ii).lme_linear.Coefficients.Estimate(2), stats(ii).lme_linear.Coefficients.SE(2));
%     linear_data.Properties.VariableNames = {'test_name', 'Growth', 'SE'};    
% 
% 
% h = bar(y, 'FaceColor', 'w', 'EdgeColor', 'k');
% errorbar(linear_data.Growth, linear_data.SE, 'kx');






end