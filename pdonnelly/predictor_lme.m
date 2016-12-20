
%% Slope Estimate Correlation Analysis - INDIVIDUAL LEVEL

s = unique(sid);
for jj = 1:length(s)
    indx = find(strcmp(s(jj), sid));
    indx = indx(isnan(score(indx)) == 0);
    tbl = table(sid(indx), long_var(indx), score(indx));
    tbl.Properties.VariableNames = {'sid', 'long_var', 'score'};
    % Rather than using a mixed linear model we want to use an ordinary least
    % squares regression. Many functions impliment this. For example
    % regress regstats or polyfit
    indiv_slopes(jj,:) = polyfit(tbl.long_var-1,tbl.score,1);
end
% reshape predictor matrix to get one score per subject
% NOTE: the argument following the original predictor matrix is the number
% of sessions involved in the analysis
% tmp = reshape(predictor, 4, numel(predictor)/4);
% % zero in on first column for unique scores
% predictor = tmp(1,:)';
% For altering predictor based on mean/median/etc
% for sub = 1:numel(predictor)
%     predictor(sub) = predictor(sub) - median(predictor);
% end
% Compute correlation
figure; hold;
[c, p] = corr(predictor, indiv_slopes(:,1),'rows','pairwise');
scatter(predictor, indiv_slopes(:,1), ifsig(predictor, indiv_slopes(:,1)));
xlabel('predictor'); ylabel('indiv slopes');
title(sprintf('%s as a predictor of %s growth rate (r=%.2f p =%.3f)', predictor_name, test_names{test}, c, p));
lsline;
% Save image
fname = sprintf('C:/Users/Patrick/Desktop/figures/LMB/%s_%s_%s_%s.eps', predictor_name, test_names{test},'growth_predictor', date);
fname2 = sprintf('C:/Users/Patrick/Desktop/figures/LMB/%s_%s_%s_%s.png', predictor_name, test_names{test},'growth_predictor', date);
print(fname, '-depsc');
print(fname2, '-dpng');
