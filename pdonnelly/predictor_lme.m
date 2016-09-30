
x = table(predictor(long_var == 1), score(long_var==2) - score(long_var==1), ...
    score(long_var==3) - score(long_var==2), score(long_var==4)-score(long_var==3), score(long_var==4)-score(long_var==1));
[R, PValue] = corrplot(x, 'varNames', {'Predictor', 'S2-S1', 'S3-S2', 'S4-S3', 'S4-S1'});

y = table(score(long_var==2) - score(long_var==1), score(long_var==3) - score(long_var==2), ...
    score(long_var==4)-score(long_var==3), score(long_var==4)-score(long_var==1), ...
    score2(long_var==2) - score2(long_var==1), score2(long_var==3) - score2(long_var==2), ...
    score2(long_var==4)-score2(long_var==3), score2(long_var==4)-score2(long_var==1));
[R, PValue] = corrplot(y, 'varNames', {'WA2-1', 'WA3-2', 'WA4-3', 'WA4-1', 'LWID2-1', 'LWID3-2', 'LWID4-3', 'LWID4-1'}, 'testR', 'on');



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


