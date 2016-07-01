function [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, time, test_name, reading_score, time_course);
% Calculates linear mixed effects on longitudinal data
% 
% [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, time, test_name, reading_score, time_course)
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
% [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, time, test_name, reading_score, time_course);

%% Create Variations for Model Testing

% Individual de-meaned data set
s = unique(sid);
% Convert reading score to matlab variable
score = cell2mat(reading_score);
% de-mean each individual's reading scores
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

%% Time Course
% Variable Selection
if time_course == 1
    long_var = hours;
elseif time_course == 2
    long_var = time;
    long_var = cell2mat(long_var);
end



% Centering of time course variable
for ii = 1:length(s)
   index = find(strcmp(s(ii),sid));
   total = 0;
   for jj = 1:length(index)
       total = plus(total, long_var(index(jj))); 
   end
   avg = total/length(index);
   
   for kk = 1:length(index);
       long_var_adj(index(kk), 1) = long_var(index(kk)) - avg;
   end
   
end
long_var = long_var_adj;

% Create squared hours variable to use in quadratic model
long_var_sq = long_var.^2;


%% Create DataSet
data_table = dataset(sid, long_var, long_var_sq, score, score_adj);

%% Calculate LME fit
% Make sid a categorical variable
data_table.sid = categorical(data_table.sid);
% Fit the model on the uncentered data as changing linearly with time
% course
lme0 = fitlme(data_table, 'score ~ long_var + (1|sid)');
% Fit the model where we predict reading_score as changing linearly with
% time course
lme = fitlme(data_table, 'score_adj ~ long_var + (1|sid)');
% Fit the model on uncentered data as changing quadratically with time
% course
lme1 = fitlme(data_table, 'score ~ long_var + long_var_sq + (1|sid)');
% Fit the model where we predict reading score as changing quadratically
% with time course
lme2 = fitlme(data_table, 'score_adj ~ long_var + long_var_sq + (1|sid)');


return


