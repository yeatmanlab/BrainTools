function [lme_linear, lme_quad, data_table] = lmeCalc(sid, long_var, score, dummyon, centering)
% Furnction: Calculates linear mixed effects on longitudinal data


s = unique(sid);
if dummyon == 0
    % Centering of time course variable
    time_adj = []; score_centered = [];
    for ii = 1:length(s)
        index = find(strcmp(s(ii),sid));
        
        for kk = 1:length(index);
            if centering == 1
                time_adj(index(kk), 1) = long_var(index(kk)) - mean(long_var(index));
            end
            if centering == 2
                score_centered(index(kk), 1) = score(index(kk)) - mean(score(index));
            end
            if centering == 3
                time_adj(index(kk), 1) = long_var(index(kk)) - mean(long_var(index));
                score_centered(index(kk), 1) = score(index(kk)) - mean(score(index));
            end
        end
    end
    uncentered = long_var;
    long_var = time_adj;
    score_adj = score_centered;

    % Create squared hours variable to use in quadratic model
    long_var_sq = long_var.^2;
    % Create DataSet
    data_table = dataset(sid, uncentered, long_var, long_var_sq, score);
    % Calculate LME fit
    % Make sid a categorical variable
    data_table.sid = categorical(data_table.sid);
  
    % Fit the model on the uncentered data as changing linearly
    lme_linear = fitlme(data_table, 'score ~ long_var + (1|sid)');
    % Fit the model on uncentered data as changing quadratically
    lme_quad = fitlme(data_table, 'score ~ long_var + long_var_sq + (1|sid)');
    
    

elseif dummyon == 1
    uncentered = long_var;      
    % recode sessions as strings
    sess_recoded = cell(size(long_var));
    for jj = 1:length(sess_recoded)
        if long_var(jj) == 0
            sess_recoded{jj} = 'Session 1.5';
        elseif long_var(jj) == 1
            sess_recoded{jj} = 'Session 1';
        elseif long_var(jj) == 2
            sess_recoded{jj} = 'Session 2';
        elseif long_var(jj) == 3
            sess_recoded{jj} = 'Session 3';
        elseif long_var(jj) == 4
            sess_recoded{jj} = 'Session 4';
        end
    end
    
    % make the longitudinal variable a categorical variable
%     long_var = categorical(long_var);
    long_var = categorical(sess_recoded);
    % Create DataSet
    data_table = dataset(sid, uncentered, long_var, score);
    % Calculate LME fit
    % Make sid a categorical variable
    data_table.sid = categorical(data_table.sid);
    % Fit the model on the centered data as changing linearly
    lme_linear = fitlme(data_table, 'score ~ long_var + (long_var|sid)');
    lme_quad = [];
end



return


