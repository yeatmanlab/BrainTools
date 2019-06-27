letterSubs = [1112;1691;1673;1887;1901;1762;1964;1676;1869;1715;1916;1951];

letterSubs = [1112;1673;1676;1869];
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

m(1,1) = mean(C.letKnowUpper_pre); se(1,1:2) = RepMeasErr([C.letKnowUpper_pre C.letKnowUpper_post]);
m(1,2) = mean(C.letKnowUpper_post); 
m(2,1) = mean(C.letKnowLower_pre); se(2,1:2) = RepMeasErr([C.letKnowLower_pre C.letKnowLower_post]);
m(2,2) = mean(C.letKnowLower_post); 
m(3,1) = mean(C.cvc_pre); se(3,1:2) = RepMeasErr([C.cvc_pre C.cvc_post]);
m(3,2) = mean(C.cvc_post); 
m(4,1) = mean(C.ppa_pre); se(4,1:2) = RepMeasErr([C.ppa_pre C.ppa_post]);
m(4,2) = mean(C.ppa_post);
m(5,1) = mean(C.evt_pre/3); se(5,1:2) = RepMeasErr([C.evt_pre/3 C.evt_post/3]);
m(5,2) = mean(C.evt_post/3); 
m(6,1) = mean(C.retell_pre); se(6,1:2) = RepMeasErr([C.retell_pre C.retell_post]);
m(6,2) = mean(C.retell_post);

[~,p1] = ttest(C.letKnowUpper_pre,C.letKnowUpper_post);
[~,p2] = ttest(C.letKnowLower_pre,C.letKnowLower_post);
[~,p3] = ttest(C.cvc_pre,C.cvc_post);
[~,p4] = ttest(C.ppa_pre,C.ppa_post);
[~,p5] = ttest(C.evt_pre,C.evt_post);
[~,p6] = ttest(C.retell_pre,C.retell_post);

c = summer;
figure;
h = errorbargraph(m',se,[],[c(20,:); c(40,:)])
ax = gca; ax.XTick = [1:6]; ax.XTickLabel = {'uppercase','lowercase','cvc','ppa','evt','retell'};
ax.XLim = [0.2 6.8];
sigstar({[.8,1.2],[1.8,2.2],[2.8,3.2],[3.8,4.2],[4.8,5.2],[5.8,6.2]},[p1,p2,p3,p4,p5,p6])
legend([h(1) h(2)],{'pre-camp','post-camp'})

