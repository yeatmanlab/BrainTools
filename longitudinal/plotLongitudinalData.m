function [m, m_demeaned, se_rm] = plotLongitudinalData(data, data_indx, sessions, column)
% Make a plot of longitudinal data
%
% [m, m_demeaned, se_rm] = plotLongitudinalData(data, usesubs, sessions, column)
%
% Inputs:
%
% data
% usesubs
% sessions
% column   - Either the column number with the behavioral data or a string
%            of the column name. Either is fine
% Outputs:
%
% Example:
%
% data =[]; usesubs = 1:4; sessions = 1:4;
% column = 'WJ_BASIC_READING_SKILLS'
% [m, m_demeaned, se_rm] = plotLongitudinalData(data, usesubs, sessions, column);

%% Argument checking
if ~exist('data','var') || isempty(data)
    [~, ~, data] =  xlsread('~/Desktop/NLR_Scores');
end
% designate which subjects and sessions to use
if ~exist('usesubs','var') || isempty(data_indx)
    data_indx = [1 2 3 4 5 6];
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
% vertcat each variable of interest
sid      = data(1:end,1);
sessnum  = vertcat(data{1:end,2});
time     = vertcat(data{1:end,4});
hours    = vertcat(data{1:end,5});
beh_data = vertcat(data{1:end,column});

%% Now plot change in the behavioral measure

% get unique subject ids
s = unique(sid);

figure; hold; %open figure
c = jet(length(data_indx)); % colormap
% allocate matrix for data with nans
m = nan(length(data_indx),length(sessions)); 

% loop over the subjects
for ii = 1:numel(data_indx)
    subSesh = find(strcmp(s(ii), sid));
    % loop over the measurements sessions for that subject
    for jj = 1:numel(subSesh)
        plot(jj, beh_data(subSesh(jj)), '-o', ...
            'markerfacecolor',c(ii,:), 'markersize',8);
        % put data into a matrix
                m(ii,jj) = beh_data(subSesh(jj));
    end
end

%% Figure in hours

figure; hold;
c = jet(length(data_indx));

m_hours = nan(length(data_indx), length(sessions));

for subj = 1:numel(data_indx)
    subSesh = find(strcmp(s(subj), sid));    
    for sesh = 1: numel(subSesh)
%        plot(hours(subSesh(sesh)), beh_data(subSesh(sesh)), '-o', ...
%         'markerfacecolor', c(subj,:), 'markersize', 8);
    m_hours(subj,sesh) = hours(subSesh(sesh));
    
    end
end

figure; hold;

plot(m_hours', m');

% format the plot nicely
colname(strfind(colname, '_')) = ' ';
ylabel(sprintf(colname)); xlabel('Hours');
grid('on')

%% Calculate means, standard error
% calculate column means
mn = nanmean(m);
% calculate mean and standard error (after de-meaning each subject)
n = length(data_indx);
m_demeaned = m - repmat(nanmean(m,2),1,length(sessions));
se_rm = nanstd(m_demeaned)./sqrt(n);
errorbar(mn,se_rm,'-ko','markerfacecolor',[0 0 0],'linewidth',2);

% format the plot nicely
colname(strfind(colname, '_')) = ' ';
ylabel(sprintf(colname)); xlabel('Session');
set(gca,'xtick',sessions);
axis('tight')
grid('on')

%% Calculate means, standard error for HOURS

mn = nanmean(m);

n = length(data_indx);
m_demeaned = m - repmat(nanmean(m,2),1,length(sessions));
se_rm = nanstd(m_demeaned)./sqrt(n);
errorbar(mn, se_rm, '-ko', 'markerfacecolor', [0 0 0], ...
    'linewidth',2);

% format the plot nicely
colname(strfind(colname, '_')) = ' ';
ylabel(sprintf(colname)); xlabel('Hours');
set(gca,'xtick',sessions);
axis('tight')
grid('on')

%% fit plot

f = fit(m_hours, m, 'lowess');
plot(f, m_hours, m);

% format the plot nicely
colname(strfind(colname, '_')) = ' ';
ylabel(sprintf(colname)); xlabel('Hours');

axis auto;
grid('on')

