% Script to calculate LME stats on reading scores

%% Select Time Course
% (1) hours, (2) days, (3) sessions
time_course = 1;

if time_course == 1
    long_var = hours;
elseif time_course == 2
    long_var = days;
elseif time_course == 3
    long_var = sessions;
end

%% Dummy coding option
% (0) off, (1) on
dummy = 0;

%% Prepare lme statistics
stats = struct; % initialize struct
% Center the long variable, if necessary
for ii = 1:length(tests.names)
    if dummy == 0        
        s = unique(sid); % find unique subjects
        % Centering of time course variable
        for ii = 1:length(s)
            index = find(strcmp(s(ii),sid));
            total = 0;
            for jj = 1:length(index)
                total = plus(total, long_var(index(jj)));
            end
            avg = total/length(index);            
            for kk = 1:length(index);
                long_var_adj(index(kk), 1) = long_var(index(kk)) - avg;
            end            
        end
        long_var = long_var_adj;
        % Create squared hours variable to use in quadratic model
        long_var_sq = long_var.^2;
        
        data_table = dataset(sid, long_var, long_var_sq, tests(ii)
    elseif dummy == 1
        
    end
    
end

%% Perform lme statistics
