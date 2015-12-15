function [m, m_demeaned, se_rm] = plotLongitudinalData(data, usesubs, sessions, column)
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
    [~, ~, data] = xlsread('/mnt/diskArray/projects/NLR/NLR_Scores.xlsx','200','A1:AO25');
end
% designate which subjects and sessions to use
if ~exist('usesubs','var') || isempty(usesubs)
    usesubs = [1 2 3 4];
end
if ~exist('sessions','var') || isempty(sessions)
    sessions = [1 2 3 4];
end

% Column can be either the name of the behavioral measure or the column
% number and we will sort this out
if isnumeric(column)
    colname  = data{1,column};
elseif ischar(column)
    colname = column;
    column = strcmp(column, data(1,:))
end
%% Pull out subject id session numbers and the desired data
sid      = data(2:end,1);
sessnum  = vertcat(data{2:end,2});
beh_data = vertcat(data{2:end,column});

%% Now plot change in the behavioral measure

% get unique subject ids
s = unique(sid);

figure; hold; %open figure
c = jet(length(usesubs)); % colormap
% allocate matrix for data with nans
m = nan(length(usesubs),length(sessions)); 

% loop over the subjects
for ii = usesubs
   idx = find(strcmp(s{ii},sid));
   % loop over the measurements sessions for that subject
   for jj = sessions
       plot(sessnum(idx(jj)),beh_data(idx(jj)), 'ko',...
           'markerfacecolor',c(ii,:), 'markersize',8);
       % put data into a matrix
       m(ii,jj) = beh_data(idx(jj));
   end
end

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