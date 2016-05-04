

%% Read Table 
% upload data to table
data_raw = readtable('~/Desktop/NLR_Scores.xlsx', 'Scores');
% upload text file of subjects as table
subs = readtable('~/LMB.dat');

% Inner join table with subjects of interest
data = innerjoin(data_raw, subs);


for ii = 1:numel(subs)
    thisSub = find(strcmp(subs(ii), T.Subject_));
    usesubs = vertcat(usesubs,thisSub);
end

% Build a new struct
data = table2struct(T(usesubs,:));



%% Argument checking
if ~exist('data','var') || isempty(data)
    [~, ~, data] =  xlsread('/home/pdonnelly/Desktop/NLR_Scores', 'Sheet2', 'A2:AO24');
end
% designate which subjects and sessions to use
if ~exist('usesubs','var') || isempty(usesubs)
    usesubs = [1 2 3 4 5 6];
end
if ~exist('sessions','var') || isempty(sessions)
    sessions = [1 2 3 4 5];
end

% Column can be either the name of the behavioral measure or the column
% number and we will sort this out
if isnumeric(column)
    colname  = data{1,column};
elseif ischar(column)
    colname = column;
    column = strcmp(colname, data(1,:));
end
%% Pull out subject id session numbers and the desired data if xlsread
sid      = data(1:end,1);
sessnum  = vertcat(data{1:end,2});
time     = vertcat(data{1:end,4});
hours    = vertcat(data{1:end,5});
beh_data = vertcat(data{1:end,column});

%% Now plot change in the behavioral measure

% get unique subject ids
s = unique(sid);

figure; hold; %open figure
c = jet(length(usesubs)); % colormap
% allocate matrix for data with nans
m = nan(length(usesubs),length(sessions)); 

% loop over the subjects
for ii = 1:numel(usesubs)
    subSesh = find(strcmp(s(ii), sid));
    % loop over the measurements sessions for that subject
    for jj = 1:numel(subSesh)
        plot(jj, beh_data(subSesh(jj)), 'ko', ...
            'markerfacecolor',c(ii,:), 'markersize',8);
        % put data into a matrix
                m(ii,jj) = beh_data(subSesh(jj));
    end
end

%% To make a separate section
% calculate column means
mn = nanmean(m);
% calculate mean and standard error (after de-meaning each subject)
n = length(usesubs);
m_demeaned = m - repmat(nanmean(m,2),1,length(sessions));
se_rm = nanstd(m_demeaned)./sqrt(n);
errorbar(mn,se_rm,'-ko','markerfacecolor',[0 0 0],'linewidth',2);

% format the plot nicely
colname(strfind(colname, '_')) = ' ';
ylabel(sprintf(colname)); xlabel('Session');
set(gca,'xtick',sessions);
axis('tight')
grid('on')

return