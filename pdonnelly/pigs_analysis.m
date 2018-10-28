%% Script for Pigs Analysis, Summer 2018
% Patrick M. Donnelly
% University of Washington
% October 27th, 2018

%% Read in Data
% run pigs_analysis script in Python notebook
% read in data from Desktop
data = readtable('C://Users/Patrick/Desktop/pigs_wordlist_data.csv');
% rename variables for ease in analysis
data.Properties.VariableNames = {'Var1', 'id', 'session', 'group', ...
    'study_name', 'word1_acc', 'word2_acc', 'pseudo1_acc', ...
    'pseudo2_acc', 'first_acc', 'second_rate', 'wj_brs', 'twre_index', ...
    'practice'};
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
model4 = 'acc ~ 1 + practice + group*session + (1-session|id)'

% focus in on dataset
real_data = data_stacked(data_stacked.type==true,:);
% run model fits
lme1 = fitlme(real_data, model1, 'FitMethod', 'ML');
lme2 = fitlme(real_data, model2, 'FitMethod', 'ML');
compare(lme1,lme2)
lme3 = fitlme(real_data, model3, 'FitMethod', 'ML');
compare(lme2,lme3)
lme4 = fitlme(real_data, model4, 'FitMethod', 'ML');
compare(lme3,lme4)

% Pseudo word analysis
% models
model1 = 'acc ~ 1 + practice + group*session + (session|id)'
model2 = 'acc ~ 1 + practice*group*session + (1-session|id)'
model3 = 'acc ~ 1 + group*session + (session|id)'


% focus in on dataset
pseudo_data = data_stacked(data_stacked.type==false,:);
% run model fits
lme1 = fitlme(pseudo_data, model1, 'FitMethod', 'ML');
lme2 = fitlme(pseudo_data, model2, 'FitMethod', 'ML');
compare(lme1,lme2)
lme3 = fitlme(pseudo_data, model3, 'FitMethod', 'ML');
compare(lme1,lme3)


%% By Group
int_data = data_stacked(data_stacked.type==true & data_stacked.group==1,:);
cntrl_data = data_stacked(data_stacked.type==true & data_stacked.group==0,:);

% models
model1 = 'acc ~ 1 + session + (session|id) + (1|acc_Indicator)'
model2 = 'acc ~ 1 + session + (session|id) + (1|acc_Indicator)'

% model fits
lme1 = fitlme(int_data, model1, 'FitMethod', 'ML')
lme2 = fitlme(cntrl_data, model2, 'FitMethod', 'ML')

%% Passage data

% acc models
model1 = 'first_acc ~ 1 + practice*group*session + (1-session|id)'
model2 = 'first_acc ~ 1 + practice*group*session + (session|id)'

% run model fits
lme_acc_1 = fitlme(data, model1, 'FitMethod', 'ML')
lme_acc_2 = fitlme(data, model2, 'FitMethod', 'ML')
compare(lme_acc_1, lme_acc_2)

% rate models
model1 = 'second_rate ~ 1 + practice*group*session + (1-session|id)'
lme_rate = fitlme(data, model1, 'FitMethod', 'ML')













