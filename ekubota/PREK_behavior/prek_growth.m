letterSubs = [1112;1691;1673;1762;1964;1676;1869;1916;1951;1715;1901;1887];

T = readtable('data.xlsx')

C = table;
C.subs = letterSubs;

for ii = 1:length(letterSubs)
    C.evt_pre(ii) = T(T.study_name == 61 & T.record_id == letterSubs(ii),:).evt_ss;
    C.evt_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).evt_ss;
    C.letKnowUpper_pre(ii) = T(T.study_name == 61 & T.record_id == letterSubs(ii),:).letknow_numlet + ...
        T(T.study_name == 61 & T.record_id == letterSubs(ii),:).letknow_numsound;
    C.letKnowUpper_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).letknow_numlet + ...
        T(T.study_name == 62 & T.record_id == letterSubs(ii),:).letknow_numsound;
    C.letKnowLower_pre(ii)= T(T.study_name == 61 & T.record_id == letterSubs(ii),:).letknow_numlowlet + ...
        T(T.study_name == 61 & T.record_id == letterSubs(ii),:).letknow_numlowsound;
    C.letKnowLower_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).letknow_numlowlet + ...
        T(T.study_name == 62 & T.record_id == letterSubs(ii),:).letknow_numlowsound;
    C.ppa_pre(ii) = T(T.study_name == 61 & T.record_id == letterSubs(ii),:).ppa_ism +...
        T(T.study_name == 61 & T.record_id == letterSubs(ii),:).ppa_fsm +...
        T(T.study_name == 61 & T.record_id == letterSubs(ii),:).ppa_pa;
    C.ppa_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).ppa_ism +...
        T(T.study_name == 62 & T.record_id == letterSubs(ii),:).ppa_fsm +...
        T(T.study_name == 62 & T.record_id == letterSubs(ii),:).ppa_pa;
    C.cvc_pre(ii)= T(T.study_name == 61 & T.record_id == letterSubs(ii),:).pals_pseudo_seta;
    C.cvc_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).pals_pseudo_seta;
    C.retell_pre(ii)= T(T.study_name == 61 & T.record_id == letterSubs(ii),:).nlmlisten_retell;
    C.retell_post(ii) = T(T.study_name == 62 & T.record_id == letterSubs(ii),:).nlmlisten_retell;
end 

m(1,1) = mean(C.letKnowUpper_pre); se(1,1) = std(C.letKnowUpper_pre)/sqrt(height(C));
m(1,2) = mean(C.letKnowUpper_post); se(1,2) = std(C.letKnowUpper_post)/sqrt(height(C));
m(1,3) = mean(C.letKnowUpper_post - C.letKnowUpper_pre); 
se(1,3) = std(C.letKnowUpper_post - C.letKnowUpper_pre)/sqrt(height(C));
m(2,1) = mean(C.letKnowLower_pre); se(2,1) = std(C.letKnowLower_pre)/sqrt(height(C));
m(2,2) = mean(C.letKnowLower_post); se(2,2) = std(C.letKnowLower_post)/sqrt(height(C));
m(2,3) = mean(C.letKnowLower_post - C.letKnowLower_pre); 
se(2,3) = std(C.letKnowLower_post - C.letKnowLower_pre)/sqrt(height(C));
m(3,1) = mean(C.cvc_pre); se(3,1) = std(C.cvc_pre)/sqrt(height(C));
m(3,2) = mean(C.cvc_post); se(3,2) = std(C.cvc_post)/sqrt(height(C));
m(3,3) = mean(C.cvc_post - C.cvc_pre); se(3,3) = std(C.cvc_post - C.cvc_pre)/sqrt(height(C));
m(4,1) = mean(C.ppa_pre); se(4,1) = std(C.ppa_pre)/sqrt(height(C));
m(4,2) = mean(C.ppa_post); se(4,2) = std(C.ppa_post)/sqrt(height(C));
m(4,3) = mean(C.ppa_post - C.ppa_pre); se(4,3) = std(C.ppa_post - C.ppa_pre)/sqrt(height(C));
m(5,1) = nanmean(C.evt_pre/3);  se(5,1) = nanstd(C.evt_pre/3)/sqrt(height(C));
m(5,2) = nanmean(C.evt_post/3);  se(5,2) = nanstd(C.evt_post/3)/sqrt(height(C));
m(5,3) = nanmean(C.evt_post/3 - C.evt_pre/3); se(5,3) = nanstd(C.evt_post/3 - C.evt_pre/3)/sqrt(height(C));
m(6,1) = mean(C.retell_pre); se(6,1) = std(C.retell_pre)/sqrt(height(C));
m(6,2) = mean(C.retell_post); se(6,2) = std(C.retell_post)/sqrt(height(C));
m(6,3) = mean(C.retell_post - C.retell_pre); se(6,3) = std(C.retell_post - C.retell_pre)/sqrt(height(C));

[~,p1] = ttest(C.letKnowUpper_pre,C.letKnowUpper_post);
d1 = computeCohen_d(C.letKnowUpper_post,C.letKnowUpper_pre,'paired');
[~,p2] = ttest(C.letKnowLower_pre,C.letKnowLower_post);
d2 = computeCohen_d(C.letKnowLower_post,C.letKnowLower_pre,'paired');
[~,p3] = ttest(C.cvc_pre,C.cvc_post);
d3 = computeCohen_d(C.cvc_post,C.cvc_pre,'paired');
[~,p4] = ttest(C.ppa_pre,C.ppa_post);
d4 = computeCohen_d(C.ppa_post,C.ppa_pre,'paired');
[~,p5] = ttest(C.evt_pre,C.evt_post);
d5 = computeCohen_d(C.evt_post,C.evt_pre,'paired');
[~,p6] = ttest(C.retell_pre,C.retell_post);
d6 = computeCohen_d(C.retell_post,C.retell_pre,'paired');

c = summer;
figure;
[h] = errorbargraph(m',se',[],[c(20,:); c(30,:); c(40,:)],.3)
ax = gca; ax.XTick = [1:6]; ax.XTickLabel = {'uppercase','lowercase','cvc','ppa','evt','retell'};
ax.XLim = [0.2 6.8];
sigstar({[.7,1],[1.7,2],[2.7,3],[3.7,4],[4.7,5],[5.7,6]},[p1,p2,p3,p4,p5,p6])
legend([h(1) h(2) h(3)],{'pre-camp','post-camp','growth'})
ylabel('Score (test dependent)')

