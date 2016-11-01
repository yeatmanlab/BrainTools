


s = unique(sid);
time_adj = []; 
for ii = 1:length(s)
    index = find(strcmp(s(ii),sid));
    for kk = 1:length(index);
        time_adj(index(kk), 1) = long_var(index(kk)) - mean(long_var(index));
    end
end
uncentered = long_var;
long_var = time_adj;

% pred_adj = [];
% pred_adj = predictor - mean(predictor);
% uncentered_predictor = predictor;
% predictor = pred_adj;

predictor_table = table(sid, long_var, predictor, score);


predictor_lme = fitlme(predictor_table, 'score ~ predictor + (1|sid)');
pred_inter_lme =fitlme(predictor_table, 'score ~ long_var + (long_var * predictor) + (long_var|sid)');


pred_anova = anova(pred_inter_lme);


score_delta = zeros(numel(score(long_var==1)), 1);
score_delta = score(long_var==4) - score(long_var==1);
figure; 
scatter(predictor(long_var == 1), score_delta);
corr(predictor(long_var == 1), score_delta);
lsline
xlabel('WASI FS2 SCORE');
ylabel('GROWTH IN READING SKILL; WJ\_BRS');
title('WASI FS2 as a predictor of Reading Growth');

%% Slope Estimate Correlation Analysis - GROUP LEVEL
% using individual slope estimates
% gather lme statistics from model
[estimates,names] = randomEffects(pred_inter_lme);
% reshape to separate slopes from estimates
tmp = reshape(estimates, 2, numel(estimates)/2);
% zero in on column with the slopes and convert to column matrix
slopes = tmp(2,:)';
% reshape predictor matrix to get one score per subject
% NOTE: the argument following the original predictor matrix is the number
% of sessions involved in the analysis
tmp = reshape(predictor, 4, numel(predictor)/4);
% zero in on first column for unique scores
predictor = tmp(1,:)';
% Compute correlation
figure; hold;
scatter(predictor, slopes);
xlabel('predictor'); ylabel('score');
lsline;


%% Slope Estimate Correlation Analysis - INDIVIDUAL LEVEL

s = unique(sid);
time_adj = []; indiv_slopes = zeros(length(s), 1);
for ii = 1:length(s)
    index = find(strcmp(s(ii),sid));
    for kk = 1:length(index);
        time_adj(index(kk), 1) = long_var(index(kk)) - mean(long_var(index));
    end
end
uncentered = long_var;
long_var = time_adj;

pred_adj = [];
pred_adj = predictor - mean(predictor);
uncentered_predictor = predictor;
predictor = pred_adj;


for jj = 1:length(s)
    indx = find(strcmp(s(jj), sid));
    tbl = table(sid(indx), long_var(indx), score(indx));
    tbl.Properties.VariableNames = {'sid', 'long_var', 'score'};
    lme = fitlme(tbl, 'score ~ long_var + (long_var|sid)');
    [estimates,names] = randomEffects(lme);
    indiv_slopes(jj, 1) = estimates(2);
end
% reshape predictor matrix to get one score per subject
% NOTE: the argument following the original predictor matrix is the number
% of sessions involved in the analysis
tmp = reshape(predictor, 4, numel(predictor)/4);
% zero in on first column for unique scores
predictor = tmp(1,:)';
% Compute correlation
figure; hold;
scatter(predictor, indiv_slopes);
xlabel('predictor'); ylabel('score');
lsline;


