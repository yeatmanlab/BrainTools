function [stats] = sessionPlot(stats)
% Creates a dummy variable-based breakdown of the influence of each 
% session to describe differing rates of change


%% Loop through tests to create Box Plots 
for ii = 1:length(stats)
    % Pull information from data struct
    test_name = char(stats(ii).test_name);
    subs = stats(ii).data_table.sid;
    sessions = stats(ii).sessions;
    score = stats(ii).data_table.score;
    % Create categories for sessions
    sessions = categorical(sessions);

    % Create figure
    figure; hold;
    % Apply labels
    title(['BoxPlot ', test_name]);
    xlabel('Session');
    ylabel('Score');
    grid('on');
    % Create box plot of sessions v. scores
    boxplot(score, sessions, 'Labels', {'1', '2', '3', '4'});
    % Save image
    fname = sprintf('~/Desktop/figures/LMB/%s-%s-%s.png','BoxPlot', strrep(test_name, '\_', '_'), date);
    print(fname, '-dpng'); 
    
    
    % Create dummy variable matrix
    dummy = dummyvar(sessions);
    % Create matrix with a column of all zeros
    n = length(score);
    x = [dummy(:,1), dummy(:,2), zeros(n,1)];
    [coeffs, confidence, r, rint, regress_stats] = regress(score, x);
    
    
    t = table(sessions, score, 'VariableNames', {'session', 'score'});
    time = [1 2]; 
    rm = fitrm(t, 'score ~ session', 'WithinDesign', time);
    
    
    
end


return