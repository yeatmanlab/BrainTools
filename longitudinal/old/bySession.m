function [stats] = sessionPlot(stats)
% Creates a dummy variable-based breakdown of the influence of each 
% session to describe differing rates of change


%% Loop through tests to create Box Plots 
for ii = 1:length(stats)
    % Pull information from data struct
    test_name = char(stats(ii).test_name);
    test_name = strrep(test_name, '\_', '_');
    subs = stats(ii).data_table.sid;
    sessions = stats(ii).sessions;
    score = stats(ii).data_table.score;
    % Create categories for sessions
    sessions = categorical(sessions);
    % Create dummy variable matrix
    dummy = dummyvar(sessions);
    % Create figure
    figure; hold;
    % Apply labels
    title(['BoxPlot ', test_name]);
    xlabel('Session');
    ylabel('Score');
    % Create box plot of sessions v. scores
    boxplot(score, sessions, 'Labels', {'1', '2', '3', '4'});
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.png','BoxPlot' test_name, date);
    print(fname, '-dpng'); 
end


return