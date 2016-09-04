%% Make figure for Flux Education neuroscience symposium Sept. 2016
% Read in data and fit model
t = readtable('NLR_Scores.xlsx');
t = t(1:198,:);
t = t(t.LMB_session==1 | t.LMB_session==2 | t.LMB_session==3 | t.LMB_session==4,:);
t.LMB_session = categorical(t.LMB_session);
lme = fitlme(t, 'WJ_BRS ~ LMB_session + (LMB_session|Subject)');
y = lme.Coefficients.Estimate; y(2:end) = y(2:end) + y(1);
e = lme.Coefficients.SE;

%% Make figure showing each individual
figure;hold('on')
sub = unique(t.Subject)
for ii = 1:length(sub)
    d = t.WJ_BRS(strcmp(t.Subject,sub{ii}));
    s = t.LMB_session(strcmp(t.Subject,sub{ii}));
    fprintf('\n%s - %d',sub{ii},length(d));
    plot(s,d,'-','color',[.5 .5 .5])
end
axis('tight');
print('BRS_ind.eps','-depsc')

%% Make figure showing group average
figure;
errorbar(y,e,'-ko', 'markerfacecolor',[.8 0 0],'markersize',15,'linewidth',3)
print('BRS_group.eps','-depsc')


