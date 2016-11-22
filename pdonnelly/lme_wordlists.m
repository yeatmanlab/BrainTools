%% LME Model for Word Lists
% Read in data and fit model
t = readtable('~/Desktop/NLR_WordLists_Scores.xlsx');
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
legend({'unspaced', 'spaced'});

%Add Error Bars
errorbar(sessions', estimates, se, '.k');
errorbar(sessions', spaced_estimates, spaced_se, '.k');



%% Adjust for Hypothesis of Individual variation in propensity
v = readtable('~/Desktop/NLR_WordLists_Scores.xlsx');
s = unique(v.Subject);
unspaced = []; spaced = []; session = [];
adjusted_sub = {}; difference = [];
spaced_count = 1;
unspaced_count = 1;
for ii = 1:length(v.Subject)
    if v.Spacing(ii) == 1
        spaced(spaced_count) = v.x4Letter(ii);
        spaced_count = spaced_count + 1;
    elseif v.Spacing(ii) == 0
        unspaced(unspaced_count) = v.x4Letter(ii);
        adjusted_sub(unspaced_count) = v.Subject(ii);
        session(unspaced_count) = v.Session(ii);
        unspaced_count = unspaced_count + 1;
    end           
end

if length(unspaced_count) ~= length(spaced_count)
    print('ERROR - UNEQUAL INFO');
end

difference = zeros(length(unspaced), 1);

for jj = 1:length(difference)
   difference(jj) = spaced(jj) - unspaced(jj); 
end

wlists = table(adjusted_sub', session', spaced', unspaced', difference);
wlists.Properties.VariableNames = {'Subject', 'Session', 'Spaced', 'Unspaced', 'Difference'};

[R, PValue] = corrplot(wlists);
lme_wl = fitlme(wlists, 'Difference ~ Session + (Session|Subject)');



one2two = corr(wlists.Difference(wlists.Session==1),wlists.Difference(wlists.Session==2))
two2three = corr(wlists.Difference(wlists.Session==2),wlists.Difference(wlists.Session==3))
three2four = corr(wlists.Difference(wlists.Session==3),wlists.Difference(wlists.Session==4))


% v.Session = categorical(v.Session);
% v.Spacing = categorical(v.Spacing);
% lme = fitlme(v, 'x4Letter ~ Session + Spacing + (Session * Spacing) + (Session|Subject)');

