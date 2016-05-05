function [] = lmeLongitudinalplot(sid, hours, test_name, reading_score, lme, lme2, data_table)
% [] = lmeLongitudinalplot(sid, hours, test_name, reading_score, lme, lme2, data_table)
% 
% Function: plots the behavioral data and overlays lme curve
% 
% Inputs:
% sid
% hours
% test_name
% reading_score
% 
% Outputs:
% 
% 
% 
% Example:
%
% data = []; subs = {'...', '...', '...'}; test_name = 'WJ_BRS';
% [sid, hours, reading_score] = prepLongitudinaldata(data, subs, ...
% test_name);
% [lme, lme2, data_table] = lmeLongitudinaldata(sid, hours, test_name, reading_score);
% [] = lmeLongitudinalplot(sid, hours, test_name, reading_score, lme, lme2, ...
% data_table);


s = unique(sid);


plot_matrix = nan(length(hours), length(reading_score)); 

for subj = 1:length(sid)
   visit_indx = find(strcmp(s(subj), sid)); 
   
   for visit = 1:length(visit_indx)
        
      
     
   end
    
end

c = jet(length(sid));

figure; hold;

% format the plot nicely
% colname(strfind(test_name, '_')) = ' ';
ylabel(test_name); xlabel('Hours');
% grid('on')







return

