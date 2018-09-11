function t = tstat_calc(m1, m2, se1, se2)
% Calculate tstat based on betas and standard errors

t = (mean(m1,2) - mean(m2,2))./sqrt(sum([se1.^2, se2.^2],2));