function [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score)
% Calculates linear mixed effects on longitudinal data
% 
% [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score)
% 
% Inputs: 
% sid
% hours
% test_name
% reading_score
% 
% Outputs:
% 
% lme
% lme2
% data_table
% 
% Example:
% 
% data = []; subs = {'...', '...', '...'}; test_name = 'WJ_BRS';
% [sid, hours, reading_score] = prepLongitudinaldata(data, subs, ...
% test_name);
% [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score);

%% Create Variations for Model Testing

% Individual de-meaned data set
s = unique(sid);
% Convert reading score to matlab variable
score = cell2mat(reading_score);
% de-mean each individual's reading scores
for ii = 1:length(s)
   index = find(strcmp(s(ii),sid));
   sum = 0;
   for jj = 1:length(index)
       sum = plus(sum, score(index(jj))); 
   end
   avg = sum/length(index);
   
   for kk = 1:length(index);
       score_sq_unique(index(kk), 1) = score(index(kk)) - avg;
   end
   
end

 
score_adj = score_sq_unique;

% Create squared hours variable to use in quadratic model
hours_sq = hours.^2;


%% Create DataSet
data_table = dataset(sid, hours, hours_sq, score, score_adj);

%% Calculate LME fit
% Make sid a categorical variable
data_table.sid = categorical(data_table.sid);
% Fit the model on the uncentered data as changing linearly with the number
% of hours of intervention
lme0 = fitlme(data_table, 'score ~ hours + (1|sid)');
% Fit the model where we predict reading_score as changing linearly with the number of
% hours of intervention
lme = fitlme(data_table, 'score_adj ~ hours + (1|sid)');
% Fit the model on uncentered data as changing quadratically with hours of
% intervention
lme1 = fitlme(data_table, 'score ~ hours + hours_sq + (1|sid)');
% Fit the model where we predict reading score as changing quadratically with hours of
% intervention
lme2 = fitlme(data_table, 'score_adj ~ hours + hours_sq + (1|sid)');


return


