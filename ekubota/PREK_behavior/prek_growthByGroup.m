letterSubs = [1112;1691;1673;1887;1901;1762;1964;1676;1869;1715;1916;1951];
languageSubs = [];

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

C.letKnowUpper_growth = C.letKnowUpper_post - C.letKnowUpper_pre;
C.letKnowLower_growth = C.letKnowLower_post - C.letKnowLower_pre;
C.cvc_growth = C.cvc_post - C.cvc_pre;
C.ppa_growth = C.ppa_post - C.ppa_pre;
C.retell_growth = C.retell_post - C.retell_pre;
C.evt_growth = C.evt_post - C.evt_pre;

D = table;
D.subs = languageSubs;

for ii = 1:length(languageSubs)
    D.evt_pre(ii) = T(T.study_name == 61 & T.record_id == languageSubs(ii),:).evt_ss;
    D.evt_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).evt_ss;
    D.letKnowUpper_pre(ii) = T(T.study_name == 61 & T.record_id == languageSubs(ii),:).letknow_numlet + ...
        T(T.study_name == 61 & T.record_id == languageSubs(ii),:).letknow_numsound;
    D.letKnowUpper_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).letknow_numlet + ...
        T(T.study_name == 62 & T.record_id == languageSubs(ii),:).letknow_numsound;
    D.letKnowLower_pre(ii)= T(T.study_name == 61 & T.record_id == languageSubs(ii),:).letknow_numlowlet + ...
        T(T.study_name == 61 & T.record_id == languageSubs(ii),:).letknow_numlowsound;
    D.letKnowLower_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).letknow_numlowlet + ...
        T(T.study_name == 62 & T.record_id == languageSubs(ii),:).letknow_numlowsound;
    D.ppa_pre(ii) = T(T.study_name == 61 & T.record_id == languageSubs(ii),:).ppa_ism +...
        T(T.study_name == 61 & T.record_id == languageSubs(ii),:).ppa_fsm +...
        T(T.study_name == 61 & T.record_id == languageSubs(ii),:).ppa_pa;
    D.ppa_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).ppa_ism +...
        T(T.study_name == 62 & T.record_id == languageSubs(ii),:).ppa_fsm +...
        T(T.study_name == 62 & T.record_id == languageSubs(ii),:).ppa_pa;
    D.cvc_pre(ii)= T(T.study_name == 61 & T.record_id == languageSubs(ii),:).pals_pseudo_seta;
    D.cvc_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).pals_pseudo_seta;
    D.retell_pre(ii)= T(T.study_name == 61 & T.record_id == languageSubs(ii),:).nlmlisten_retell;
    D.retell_post(ii) = T(T.study_name == 62 & T.record_id == languageSubs(ii),:).nlmlisten_retell;
end 

D.letKnowUpper_growth = D.letKnowUpper_post - D.letKnowUpper_pre;
D.letKnowLower_growth = D.letKnowLower_post - D.letKnowLower_pre;
D.cvc_growth = D.cvc_post - D.cvc_pre;
D.ppa_growth = D.ppa_post - D.ppa_pre;
D.retell_growth = D.retell_post - D.retell_pre;
D.evt_growth = D.evt_post - D.evt_pre;

m(1,1) = mean(C.letKnowUpper_growth); 
se(1,1) = std(C.letKnowUpper_growth)/sqrt(height(C));
m(1,2) = mean(D.letKnowUpper_growth);
se(1,2) = std(D.letKnowUpper_growth)/sqrt(height(D));
m(2,1) = mean(C.letKnowLower_growth); 
se(2,1) = std(C.letKnowLower_growth)/sqrt(height(C));
m(2,2) = mean(D.letKnowLower_growth); 
se(2,2) = std(D.letKnowLower_growth)/sqrt(height(D));
m(3,1) = mean(C.cvc_growth);
se(3,1) = std(C.cvc_growth)/sqrt(height(C));
m(3,2) = mean(D.cvc_growth); 
se(3,2) = std(D.cvc_growth)/sqrt(height(D));
m(4,1) = mean(C.ppa_growth); 
se(4,1) = std(C.ppa_growth)/sqrt(height(C));
m(4,2) = mean(D.ppa_growth);
se(4,2) = std(D.ppa_growth)/sqrt(height(D));
m(5,1) = mean(C.evt_growth); 
se(5,1) = std(C.evt_growth)/sqrt(height(C));
m(5,2) = mean(D.evt_growth); 
se(5,2) = std(D.evt_growth)/sqrt(height(D));
m(6,1) = mean(C.retell_growth);
se(6,1) = std(C.retell_growth)/sqrt(height(C));
m(6,2) = mean(D.retell_growth);
se(6,2) = std(D.retell_growth)/sqrt(height(D));


[~,p1] = ttest2(C.letKnowUpper_growth,D.letKnowUpper_growth);
[~,p2] = ttest2(C.letKnowLower_growth,D.letKnowLower_growth);
[~,p3] = ttest2(C.cvc_growth,D.cvc_growth);
[~,p4] = ttest2(C.ppa_growth,D.ppa_growth);
[~,p5] = ttest2(C.evt_growth,D.evt_growth);
[~,p6] = ttest2(C.retell_growth,D.retell_growth);

c = summer;
figure;
h = errorbargraph(m',se,[],[c(20,:); c(40,:)])
ax = gca; ax.XTick = [1:6]; ax.XTickLabel = {'uppercase','lowercase','cvc','ppa','evt','retell'};
ax.XLim = [0.2 6.8];
sigstar({[.8,1.2],[1.8,2.2],[2.8,3.2],[3.8,4.2],[4.8,5.2],[5.8,6.2]},[p1,p2,p3,p4,p5,p6])
legend([h(1) h(2)],{'letter','language'})

