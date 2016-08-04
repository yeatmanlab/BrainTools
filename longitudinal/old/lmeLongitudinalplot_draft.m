function [sub_mat, s] = lmeLongitudinalplot(subs, sid, hours, test_names, reading_score, stats)
% [] = lmeLongitudinalplot(sid, hours, test_name, reading_score, stats);
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


for ii = 1:length(test_names)

plot_table = table(sid, hours, reading_score);



s = unique(sid);

% 
plot_matrix = nan(length(hours), length(s)); 

figure; hold;
% c = jet(length(s));
for subj = 1:length(s)
   visit_indx = find(strcmp(s(subj), sid));
    sub_mat = [];
   for visit = 1:length(visit_indx)
       sub_mat(visit, 1) = plot_table.hours(visit_indx(visit));
       sub_mat(visit, 2) = plot_table.reading_score{visit_indx(visit)};
%         plot(plot_table.hours(visit_indx(visit)), plot_table.reading_score{visit_indx(visit)}, '-o');
   end 
   plot(sub_mat(:,1), sub_mat(:,2),'-o');
end


% format the plot nicely
% colname(strfind(test_name, '_')) = ' ';
ylabel(test_name); xlabel('Hours'); legend(subs);
grid('on')

%% Add in the best fit line
xx = [0, 200];
y = polyval(flipud(lme.Coefficients.Estimate),xx);
plot(xx,y,'--k','linewidth',2);


%% Plot Centered


return

