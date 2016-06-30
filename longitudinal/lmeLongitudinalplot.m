function [stats] = lmeLongitudinalplot(stats, test_names, subs, time_course);
% [stats] = lmeLongitudinalplot(stats, test_names, subs);
% 
% Function: plots the behavioral data and overlays lme curve
% 
% Inputs:
% stats: in the form of a struct
% 
% Outputs:
% 
% 
% 
% Example:
%
%% Variables
raveo_time = [0 14 24];
raveo_hours = [0 25 50 75 100 125 150];
lmb_time = [0 20 40 60 80 100];
lmb_hours = [0 50 100 150 200];

if time_course == 1
    x_name = 'hours';
    % time course for hours
    % rave-o
    xx = raveo_hours;

    % % lmb
    % xx = lmb_hours;

elseif time_course == 2
    x_name = 'days';
    % time course for days
    % rave-o
    xx = raveo_time;
    % lmb
    % xx = lmb_time;

end

%% Plot Data
% for ii = 1:length(test_names)
%     % Create table for individual test
%     plot_table = table(stats(ii).data_table.sid, stats(ii).data_table.long_var, ...
%         stats(ii).data_table.score);
%     % Name variables in plot table
%     plot_table.Properties.VariableNames = {'sid', x_name, 'score'};
%     % find the number of individual subjects
%     s = unique(stats(ii).data_table.sid);
%     % Create plot matrix
%     plot_matrix = nan(length(plot_table.sid), length(s)); 
% 
%     % intialize and fix figure    
%     figure; hold;
%     % loop over each subject over sessions
%     for subj = 1:length(s)
%          % find the sessions for each subject
%          visit_indx = find(plot_table.sid == s(subj));
%          % create empty array for individual subject 
%          sub_mat = [];
%          % loop over each visit
%          for visit = 1:length(visit_indx)
%              % assign session values to plot array
%              if time_course == 1
%                  sub_mat(visit, 1) = plot_table.hours(visit_indx(visit));
%                  sub_mat(visit, 2) = plot_table.score(visit_indx(visit));
%              elseif time_course == 2
%                  sub_mat(visit, 1) = plot_table.days(visit_indx(visit));
%                  sub_mat(visit, 2) = plot_table.score(visit_indx(visit)); 
%              end
%          end
%          % plot the scores, hours v. score
%          plot(sub_mat(:,1), sub_mat(:,2),'-o');
%     end
% 
%     % format the plot nicely
%     ylabel(test_names(ii)); xlabel(x_name); 
%     legend(subs, 'Location', 'eastoutside');
%     title([test_names(ii), '.uncentered']);
%     
%     grid('on')
% 
%     % Add line of best fit
%     low = min(stats(ii).data_table.long_var);
%     high = max(stats(ii).data_table.long_var);
%     xx = [low low/2 0 high/2 high];
%     y = polyval(flipud(stats(ii).lme1.Coefficients.Estimate),xx);
%     plot(xx,y,'--k','linewidth',2);
%     
%      % Add p value for best fit line
%     text(high, 0, ['P Value: ', double(stats(ii).lme.Coefficients.pValue(2))]); 
%     
% end

% Plot Centered
for ii = 1:length(test_names)
    % Create table for individual test
    plot_table = table(stats(ii).data_table.sid, stats(ii).data_table.long_var, ...
         stats(ii).data_table.score_adj);
    % Name variables in plot table
    plot_table.Properties.VariableNames = {'sid', x_name, 'score'};
    % find the number of individual subjects
    s = unique(stats(ii).data_table.sid);
    % Create plot matrix
    plot_matrix = nan(length(plot_table.sid), length(s)); 

    % intialize and fix figure    
    figure; hold;
    % loop over each subject over sessions
    for subj = 1:length(s)
         % find the sessions for each subject
         visit_indx = find(plot_table.sid == s(subj));
         % create empty array for individual subject 
         sub_mat = [];
         % loop over each visit
         for visit = 1:length(visit_indx)
             % assign session values to plot array
             if time_course == 1
                 sub_mat(visit, 1) = plot_table.hours(visit_indx(visit));
                 sub_mat(visit, 2) = plot_table.score(visit_indx(visit));
             elseif time_course == 2
                 sub_mat(visit, 1) = plot_table.days(visit_indx(visit));
                 sub_mat(visit, 2) = plot_table.score(visit_indx(visit)); 
             end
         end
         % plot the scores, hours v. score
         plot(sub_mat(:,1), sub_mat(:,2),'-o');
    end

    % format the plot nicely
    ylabel(test_names(ii)); xlabel(x_name); 
    legend(subs, 'Location', 'westoutside');
    title([test_names(ii), 'vs ', x_name]);
    grid('on')

    % Add linear line of best fit
    low = min(stats(ii).data_table.long_var);
    high = max(stats(ii).data_table.long_var);
    xx = [low low/2 0 high/2 high];
    y = polyval(flipud(stats(ii).lme.Coefficients.Estimate),xx);
    plot(xx,y,'--k','linewidth',2);
    
    % Add shaded error bar
    err = repmat(stats(ii).lme.Coefficients.SE(2), 1, length(xx));
    shadedErrorBar(xx, y, err);
    
    % Add p value for best fit line
    p_linear = double(stats(ii).lme.Coefficients.pValue(2));
    text(-40, 8, num2str(p_linear), 'FontSize', 12); 
    
    % Add quadratic line of best fit
    y = polyval(flipud(stats(ii).lme2.Coefficients.Estimate),xx);
    plot(xx,y,'--b','linewidth',2);
    
     % Add shaded error bar
    err = repmat(stats(ii).lme2.Coefficients.SE(3), 1, length(xx));
    shadedErrorBar(xx, y, err);
    
    % Add p value for best fit line
    p_quad = double(stats(ii).lme2.Coefficients.pValue(3));
    text(-40, 6, num2str(p_quad), 'Color', 'blue', 'FontSize', 12);

end



% 
% % format the plot nicely
% % colname(strfind(test_name, '_')) = ' ';
% ylabel(test_name); xlabel('Hours'); title('LMB pilot 6'); legend(subs);
% grid('on')

% %% Add in the best fit line
% xx = [0, 60, 120];
% y = polyval(flipud(lme.Coefficients.Estimate),xx);
% plot(xx,y,'--k','linewidth',2);







return

