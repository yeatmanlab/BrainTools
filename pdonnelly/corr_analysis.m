
 x = 'predictor'; y = test_name{1};

one_1to2 = score(uncentered == 2) - score(uncentered == 1);
one_2to3 = score(uncentered == 3) - score(uncentered == 2); 
one_3to4 = score(uncentered == 4) - score(uncentered == 3); 
one_1to4 = score(uncentered == 4) - score(uncentered == 1);
% 
% two_1to2 = score2(long_var == 2) - score2(long_var == 1); 
% two_2to3 = score2(long_var == 3) - score2(long_var == 2);  
% two_3to4 = score2(long_var == 4) - score2(long_var == 3);  
% two_1to4 = score2(long_var == 4) - score2(long_var == 1); 

figure; hold;

subplot(2,2,1);
scatter(predictor, one_1to2, ifsig(predictor, one_1to2));
lsline;
xlabel(sprintf('%s init', x));
ylabel(sprintf('%s init', y));

subplot(2,2,2);
scatter(predictor, one_2to3, ifsig(predictor, one_2to3));
lsline;
xlabel(sprintf('%s init', x));
ylabel(sprintf('%s mid', y));

subplot(2,2,3);
scatter(predictor, one_3to4, ifsig(predictor, one_3to4));
lsline;
xlabel(sprintf('%s init', x));
ylabel(sprintf('%s end', y));

subplot(2,2,4);
scatter(predictor, one_1to4, ifsig(predictor, one_1to4));
lsline;
xlabel(sprintf('%s init', x));
ylabel(sprintf('%s total', y));






% x = table(predictor(long_var == 1), score(long_var==2) - score(long_var==1), ...
%     score(long_var==3) - score(long_var==2), score(long_var==4)-score(long_var==3), score(long_var==4)-score(long_var==1));
% [R, PValue] = corrplot(x, 'varNames', {'Predictor', 'S2-S1', 'S3-S2', 'S4-S3', 'S4-S1'});
% 
% y = table(score(long_var==2) - score(long_var==1), score(long_var==3) - score(long_var==2), ...
%     score(long_var==4)-score(long_var==3), score(long_var==4)-score(long_var==1), ...
%     score2(long_var==2) - score2(long_var==1), score2(long_var==3) - score2(long_var==2), ...
%     score2(long_var==4)-score2(long_var==3), score2(long_var==4)-score2(long_var==1));
% [R, PValue] = corrplot(y, 'varNames', {'WA2-1', 'WA3-2', 'WA4-3', 'WA4-1', 'LWID2-1', 'LWID3-2', 'LWID4-3', 'LWID4-1'}, 'testR', 'on');
% 
