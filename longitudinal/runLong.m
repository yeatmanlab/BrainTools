function [lme, lme2, data_table] = runLong()
% runLong
% 
% 
% function: runs the battery of longitudinal functions in the lme sequence
% 
% Inputs:
% 
% data
% subs
% test_name
% 
% Outputs:
% 
% 
% 
% [lme, lme2, data_table] = runLong();

data = [];

subs = input('What subjects will you be using? Enter as {..., ..., ...} ');

test_name = input('What test will you be using? ');

[sid, hours, test_name, reading_score] = prepLongitudinaldata(data, subs, test_name);
[lme, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score);
[sub_mat, s] = lmeLongitudinalplot(subs, sid, hours, test_name, reading_score, lme, lme2, data_table);









return