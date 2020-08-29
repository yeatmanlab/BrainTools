%% Script for Pigs Analysis, Summer 2018
% Patrick M. Donnelly
% University of Washington
% October 27th, 2018

%% Read in Data
% run pigs_analysis script in Python notebook
% read in data from Desktop
data = readtable('C://Users/donne/Desktop/pigs_wordlist_data.csv');
%data = readtable('~/Desktop/pigs_wordlist_data.csv');
% rename variables for ease in analysis
data.Properties.VariableNames = {'Var1', 'id', 'session', 'group', ...
    'study_name', 'word1_acc', 'word2_acc', 'pseudo1_acc', ...
    'pseudo2_acc', 'first_acc', 'second_rate', 'wj_brs', 'twre_index', ...
    'ctopp_rapid','practice'};
% make time and group variables categorical
categorical(data.session);
categorical(data.group);
%data_stacked.wj_brs = zscore(data_stacked.wj_brs);
% calculate difference scores and create new variables
data.worddiff = data.word2_acc - data.word1_acc;
data.pseudodiff = data.pseudo2_acc - data.pseudo1_acc;
%extend practice variable to both visits for LME analysis
for sub = 1:length(data.id)
   if isnan(data.practice(sub))
       data.practice(sub) = data.practice(sub+1);
   end
end
% create new stacked dataset for wordlist data analysis
data_stacked = stack(data,{'word1_acc','word2_acc',...
    'pseudo1_acc','pseudo2_acc'},'NewDataVariableName','acc');
data_stacked.type = data_stacked.acc_Indicator == 'word1_acc' | ...
    data_stacked.acc_Indicator=='word2_acc';

%% LME analysis {preregistered}

%% Real word analysis
% models
model1 = 'acc ~ 1 + practice + group*session + (1|id)';
model2 = 'acc ~ 1 + session*practice*group + (1|id) + (1|acc_Indicator)';
model3 = 'acc ~ 1 + practice*group*session + (1-session|id)+ (1|acc_Indicator)';
model4 = 'acc ~ 1 + practice + group*session + (1-session|id)';
model5 = 'acc ~ 1 + wj_brs + twre_index + ctopp_rapid + practice*group*session + (1-session|id)+ (1|acc_Indicator)';

% focus in on dataset
real_data = data_stacked(data_stacked.type==true,:);
% run model fits
lme1 = fitlme(real_data, model1, 'FitMethod', 'ML');
lme2 = fitlme(real_data, model2, 'FitMethod', 'ML');
compare(lme1,lme2)
lme3 = fitlme(real_data, model3, 'FitMethod', 'ML'); %best model
compare(lme2,lme3)
lme4 = fitlme(real_data, model4, 'FitMethod', 'ML');
compare(lme3,lme4)
lme5 = fitlme(real_data, model5, 'FitMethod', 'ML');

% Pseudo word analysis
pseudo_data = data_stacked(data_stacked.type==false,:);
% models
model1 = 'acc ~ 1 + practice + group*session + (session|id)' % best model
model2 = 'acc ~ 1 + practice*group*session + (1-session|id)'
model3 = 'acc ~ 1 + group*session + (session|id)'
model4 = 'acc ~ 1 + wj_brs + twre_index + ctopp_rapid + practice + group*session + (session|id)';



% run model fits
lme1 = fitlme(pseudo_data, model1, 'FitMethod', 'ML');
lme2 = fitlme(pseudo_data, model2, 'FitMethod', 'ML');
compare(lme1,lme2)
lme3 = fitlme(pseudo_data, model3, 'FitMethod', 'ML');
compare(lme1,lme3)
lme4 = fitlme(pseudo_data, model4, 'FitMethod', 'ML');


%% By Group
int_data = data_stacked(data_stacked.group==1,:);
cntrl_data = data_stacked(data_stacked.group==0,:);
real_int = int_data(int_data.type==true,:);
real_control = cntrl_data(cntrl_data.type==true,:);
pseudo_int = int_data(int_data.type==false,:);
pseudo_control = cntrl_data(cntrl_data.type==false,:);

% models
model1 = 'second_rate ~ 1 + session + (session|id)'
model2 = 'acc ~ 1 + session + (session|id) + (1|acc_Indicator)'

% model fits
lme1 = fitlme(int_data, model1, 'FitMethod', 'ML')
lme2 = fitlme(cntrl_data, model2, 'FitMethod', 'ML')

% model fits for intervention/control only for real/pseudo
lme_realint = fitlme(real_int, model2, 'FitMethod', 'ML')
lme_realcontrol = fitlme(real_control, model2, 'FitMethod', 'ML')
lme_pseudoint = fitlme(pseudo_int, model2, 'FitMethod', 'ML')
lme_pseudocontrol = fitlme(pseudo_control, model2, 'FitMethod', 'ML')

% models
model1 = 'acc ~ 1 + group*session + (1|id)';
model2 = 'acc ~ 1 + session*group + (1|id) + (1|acc_Indicator)';
model3 = 'acc ~ 1 + group*session + (1-session|id)+ (1|acc_Indicator)';
model4 = 'acc ~ 1 + group*session + (1-session|id)';
model5 = 'acc ~ 1 + wj_brs + twre_index + ctopp_rapid + group*session + (1-session|id)+ (1|acc_Indicator)';
% run model fits
lme1 = fitlme(int_data, model1, 'FitMethod', 'ML');
lme2 = fitlme(int_data, model2, 'FitMethod', 'ML');
compare(lme1,lme2)
lme3 = fitlme(int_data, model3, 'FitMethod', 'ML'); %best model
compare(lme2,lme3)
lme4 = fitlme(int_data, model4, 'FitMethod', 'ML');
compare(lme3,lme4)
lme5 = fitlme(int_data, model5, 'FitMethod', 'ML');

%% Passage data

% acc models
model1 = 'first_acc ~ 1 + practice*group*session + (1-session|id)'
model2 = 'first_acc ~ 1 + practice*group*session + (session|id)'

% run model fits
lme_acc_1 = fitlme(data, model1, 'FitMethod', 'ML') %best model
lme_acc_2 = fitlme(data, model2, 'FitMethod', 'ML')
compare(lme_acc_1, lme_acc_2)

% rate models
model1 = 'second_rate ~ 1 + practice*group*session + (1-session|id)'
lme_rate = fitlme(data, model1, 'FitMethod', 'ML')




%intevention and control group only
int_data = int_data(~isnan(int_data.first_acc))

acc_model = 'first_acc ~ 1 + session + (session|id)'
lme_acc_int = fitlme(int_data, acc_model, 'FitMethod', 'ML')
lme_acc_control = fitlme(cntrl_data, acc_model, 'FitMethod', 'ML')

rate_model = 'second_rate ~ 1 + session + (session|id)'
lme_rate_int = fitlme(int_data, rate_model, 'FitMethod', 'ML')
lme_rate_control = fitlme(cntrl_data, rate_model, 'FitMethod', 'ML')








