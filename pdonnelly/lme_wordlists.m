%% LME Model for Word Lists
% Read in data and fit model
t = readtable('C:\Users\donne\Desktop\NLR_Scores_WordLists.xlsx');
t.Session = categorical(t.Session);
t.Spacing = categorical(t.Spacing);
lme = fitlme(t, 'x4Letter ~ Session + Spacing + (Session * Spacing) + (Session|Subject)');

%% with ANOVA testing
lme_b = fitlme(t, 'x4Letter ~ Session + Spacing + (Session * Spacing) + (Session|Subject)', 'DummyVarCoding', 'effects');
lme_anova = anova(lme_b);

%% Make figure showing Group Growth
figure; hold;
%Plot Individual Trends
sessions = [1 2 3 4];
num_sess = length(sessions);
%UnSpaced data
estimates = zeros(num_sess, 1);
se = zeros(num_sess, 1);
p = zeros(num_sess, 1);
%Spaced data
spaced_estimates = zeros(num_sess, 1);
spaced_se = zeros(num_sess, 1);
spaced_p = zeros(num_sess, 1);
%
for num = 1:num_sess
    estimates(num, 1) = lme.Coefficients.Estimate(num);
    se(num, 1) = lme.Coefficients.SE(num);
    p(num, 1) = round(lme.Coefficients.pValue(num), 3);
    
    spaced_estimates(num, 1) = lme.Coefficients.Estimate(num + 4);
    spaced_se(num, 1) = lme.Coefficients.SE(num + 4);
    spaced_p(num, 1) = round(lme.Coefficients.pValue(num + 4), 3);
end
spaced_estimates(1,1) = estimates(1,1) + spaced_estimates(1,1);
for num = 2:num_sess
    estimates(num, 1) = (estimates(1,1) + estimates(num, 1));
%     %optionA
%     spaced_estimates(num, 1) = (spaced_estimates(1,1) + spaced_estimates(num, 1));
    %optionB
    spaced_estimates(num, 1) = (estimates(num, 1) + spaced_estimates(num, 1));    
end
h = plot(sessions', estimates, sessions', spaced_estimates);

%Format Plot
ax = gca;   
ax.XLim = [0.5000 4.5000];
ax.YLim = [0 (max(estimates) + 5)];
ax.XAxis.TickValues = [1 2 3 4];
xlabel('Session'); ylabel('Score');
title('Pseudoword LME Mean Growth Model');
grid('on');
axis('tight');


%Add Error Bars
errorbar(sessions', estimates, se, '.k');
errorbar(sessions', spaced_estimates, spaced_se, '.k');


