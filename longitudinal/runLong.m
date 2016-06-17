function [stats] = runLong()
% runLong
% 
% 
% function: runs the battery of longitudinal functions in the lme sequence
% 
% Inputs:
% 
% data
% subs
% test_names
% 
% Outputs:
% 
% 
% 
% [stats] = runLong();

data = [];
stats = struct;
subs = {'201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM', '152_TC'};
test_names = {'WJ_BRS', 'WJ_RF', 'TWRE_INDEX'};

% subs = input('What subjects will you be using? Enter as {..., ..., ...} ');
% test_names = input('What tests will you be using? Enter as {..., ..., ...} '); 



% Gather data and perform statistics
for ii = 1:length(test_names)
   
    test_name = test_names(ii);
        
    [sid, hours, test_name, reading_score] = prepLongitudinaldata(data, subs, test_name);
    [lme0, lme, lme1, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score);
   
    stats(ii).test_name = test_name;
    stats(ii).lme0 = lme0;
    stats(ii).lme = lme;
    stats(ii).lme1 = lme1;
    stats(ii).lme2 = lme2;
    stats(ii).data_table = data_table;
    
end

% Plot data
[subs] = lmeLongitudinalplot(stats, test_names, subs);



return