function [lme_linear, lme_quad, data_table] = lmeLongitudinaldata(sid, time, score)
% Furnction: Calculates linear mixed effects on longitudinal data
% 
% Inputs: 
% 
% Outputs:
% 
% Example:
% [lme_linear, lme_quad, data_table] = lmeLongitudinaldata(sid, time, score);

%% Create Variations for Model Testing

% Center each individual's reading scores
s = unique(sid);
for ii = 1:length(s)
   index = find(strcmp(s(ii),sid));
   total = 0;
   for jj = 1:length(index)
       total = plus(total, score(index(jj))); 
   end
   avg = total/length(index);
   
   for kk = 1:length(index);
       score_sq_unique(index(kk), 1) = score(index(kk)) - avg;
   end
end
score_adj = score_sq_unique;


% Centering of time course variable
for ii = 1:length(s)
   index = find(strcmp(s(ii),sid));
   total = 0;
   for jj = 1:length(index)
       total = plus(total, time(index(jj))); 
   end
   avg = total/length(index);
   
   for kk = 1:length(index);
       time_adj(index(kk), 1) = time(index(kk)) - avg;
   end
   
end
time = time_adj;

% Create squared hours variable to use in quadratic model
time_sq = time.^2;


%% Create DataSet
data_table = dataset(sid, time, time_sq, score, score_adj);

%% Calculate LME fit
% Make sid a categorical variable
data_table.sid = categorical(data_table.sid);
% Fit the model on the uncentered data as changing linearly with time
% course
lme_linear = fitlme(data_table, 'score ~ long_var + (1|sid)');
% Fit the model where we predict reading_score as changing linearly with
% time course
% lme = fitlme(data_table, 'score_adj ~ long_var + (1|sid)');
% Fit the model on uncentered data as changing quadratically with time
% course
lme_quad = fitlme(data_table, 'score ~ long_var + long_var_sq + (1|sid)');
% Fit the model where we predict reading score as changing quadratically
% with time course
% lme2 = fitlme(data_table, 'score_adj ~ long_var + long_var_sq + (1|sid)');


%% Fit Logistic Growth Function

% [B, dev, logistic_stats] = mnrfit(long_var_sq, score); 

return


