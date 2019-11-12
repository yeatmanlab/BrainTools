T = readtable('/Users/emilykubota/Desktop/race.csv');
subs = unique(T.record_id);

for ii = 1:length(subs)
    C(ii,:) = T(T.record_id == subs(ii) & strcmp(T.redcap_event_name,'subject_intake_arm_1'),19:27)
end 

X = table;
X.white = C.race___1;
X.black = C.race___2;
X.asian = C.race___3;
X.native_american = C.race___4;
X.pacific_islander = C.race___5;
X.other = C.race___98;
X.no_answer = C.race___99;
X.ethnicity = C.ethnicity;

nWhite = sum(X.white)
nBlack = sum(X.black)
nAsian = sum(X.asian)
nNativeAmerican = sum(X.native_american)
nPacificIslander = sum(X.pacific_islander)
nOther = sum(X.other)
nHispanic = height(X(X.ethnicity == 1,:))