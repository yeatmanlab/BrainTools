%Script for Pigs Analysis, Summer 2018
%Patrick M Donnelly
%University of Washington
%September 11, 2018

%% Read in Data
% Run pigs_analysis script in Python notebook

data = readtable('C://Users/Patrick/Desktop/pigs_wordlist_data.csv');
categorical(data.int_session);
categorical(data.pigs_casecontrol);
data.worddiff = data.pigs_word2_acc - data.pigs_word1_acc;
data.pseudodiff = data.pigs_pseudo2_acc - data.pigs_pseudo1_acc;


data2 = stack(data,{'pigs_word1_acc','pigs_word2_acc','pigs_pseudo1_acc','pigs_pseudo2_acc'},'NewDataVariableName','acc')
data2.type = data2.acc_Indicator=='pigs_word1_acc' | data2.acc_Indicator=='pigs_word2_acc';
%% LME Model
word_simple_model = 'word_acc ~ 1 + int_session + pigs_casecontrol + (1|record_id)'
word_model = 'word_acc ~ 1 + pigs_casecontrol*int_session + (1|record_id)'
word_lme = fitlme(data, word_model, 'FitMethod', 'ML') 

pseudo_model = 'pseudo_acc ~ 1 + pigs_casecontrol*int_session + (1|record_id)'
pseudo_lme = fitlme(data, pseudo_model, 'FitMethod', 'ML') 

%% 

% Real words
lme1 = fitlme(data2(data2.type==true,:), 'acc ~ 1 + wj_brs + pigs_practice_numstories + pigs_casecontrol*int_session + (1|record_id)', 'FitMethod', 'ML')
lme2 = fitlme(data2(data2.type==true,:), 'acc ~ 1 + pigs_practice_numstories * pigs_casecontrol*int_session + (1|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML')
compare(lme1,lme2)
lme3 = fitlme(data2(data2.type==true,:), 'acc ~ 1 + pigs_casecontrol*int_session + (int_session|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML')
compare(lme2,lme3)

% Pseudo
lme1 = fitlme(data2(data2.type==false,:), 'acc ~ 1 + pigs_practice_numstories*pigs_casecontrol*int_session + (1|record_id)', 'FitMethod', 'ML')

data2.int_session = categorical(data2.int_session);
data2.pigs_casecontrol = categorical(data2.pigs_casecontrol);
data2.wj_brs = zscore(data2.wj_brs);
data2.pigs_practice_numstories = zscore(data2.pigs_practice_numstories);
lme2 = fitlme(data2(data2.type==false,:), 'acc ~ 1 + pigs_practice_numstories*pigs_casecontrol*int_session + (1|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML'...
)
compare(lme1,lme2)
lme3 = fitlme(data2(data2.type==false,:), 'acc ~ 1 + pigs_casecontrol*int_session + (int_session|record_id)', 'FitMethod', 'ML')
compare(lme1,lme3)

% all
lme_all = fitlme(data2, 'acc ~ 1 + pigs_casecontrol*int_session + (int_session|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML')

%% Just the intervention group
lme4 = fitlme(data2(data2.type==true & data2.pigs_casecontrol==1,:), 'acc ~ 1 + int_session + (int_session|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML')
lme5 = fitlme(data2(data2.type==true & data2.pigs_casecontrol==0,:), 'acc ~ 1 + int_session + (int_session|record_id) + (1|acc_Indicator)', 'FitMethod', 'ML')

