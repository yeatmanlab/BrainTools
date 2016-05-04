function HCP_statsRobust(afq, test, numGroups, boundary, diffProp, tractNum, tractRange)
% This should provide the opportunity to run a multitude of statistical
% operations on any selection of the data. Only required arguments are afq
% and test. To exclude intermediary arguments but define later arguments,
% include [] in place of undefined variables. Defaults defined below.

% afq is the afq structure. test is the behavioral measure of interest from
% the metadata field. numGroups is the number of groups to be classified.
% boundary should be entered as a column vector of boundaries equal to
%   numGroups - 1.
% diffProp is a cell array of strings.
% tractNum can accept any number of tracts of interest
% tractRange specifies the range of points along a specified tract to
%   analyze.

% example: 
% afq = afq;
% test = 'ReadEng_AgeAdj';
% numGroups = 3;
% boundary = [80 120];
% diffProp = {'fa', 'md'};
% tractNum = [13 19];
% tractRange = [45:70];
% HCP_statsRobust(afq, test, numGroups, diffProp, boundary, tractNum, tractRange)
% (In this example, we will work to classify the results into 3 groups
% based on age adjusted reading skills with boundaries at 80 and 120, using
% the fa and md values of the Left ILF and Left Arcuate in the 26 points
% from 45:70 along each tract.)


%% Set defaults for optional arguments

if nargin < 2
    sprintf('%s', 'Requires at least 2 arguments. Type help HCP_statsRobust for more info');
    return;
end

% Define scores for test
if iscell(test) % for future support of multiple tests
    score = cell(1, numel(test));
    for ii = 1:numel(test)
        score{ii} = AFQ_get(afq, 'metadata', test{ii});
    end
else
    score = AFQ_get(afq, 'metadata', test);
end

if nargin < 3 || isempty(numGroups)
    numGroups = 1;
end

if nargin < 4 || isempty(boundary)
    if numGroups == 1;
        boundary = 0;
    elseif numGroups == 2;
        boundary = nanmean(score);
    elseif numGroups == 3;
        boundary = [nanmean(score)-nanstd(score) nanmean(score)-nanstd(score)];
    elseif numGroups == 4;
        sprintf('%s', 'Please provide boundaries for more than 3 groups');
        return;
    end
end

if nargin < 5 || isempty(diffProp)
    diffProp = {'fa', 'md'};
end

if nargin < 6 || isempty(tractNum)
    tractNum = 1:20;
end

if nargin < 7 || isempty(tractRange)
    tractRange = 1:100;
end

%% Set up parameters for analyses

    fgnames = AFQ_get(afq, 'fgnames');

for ii = 1:numel(tractNum)
    tract = tractNum(ii);
    if ischar(diffProp)
        assignin('caller', diffProp, afq.vals.(diffProp){tract}(:, tractRange));
    elseif iscell(diffProp)
        for jj = 1:numel(diffProp)
        assignin('caller', strcat(diffProp{jj}, num2str(tract)), afq.vals.(diffProp{jj}){tract}(:, tractRange));
        end
    end
end
    
switch numGroups % expand more later
    case 1
    case 2
        poor = find(score <= boundary);
        good = find(score > boundary);
    case 3
        poor = find(score <= boundary(1));
        average = find(score > boundary(1) & score < boundary(2));
        good = find(score >= boundary(2));
end

    
%     % plot them out
%     figure;
%     subplot(1,2,1);
%     plot(mean(fa, 2), r, 'ko', 'markerfacecolor', [.8 0 0]);lsline
%     xlabel(sprintf('%s (tract num %d) FA', fgnames{fgnum}, fgnum),...
%         'fontsize', 14, 'fontname', 'times')
%     ylabel('Reading score','fontsize', 14, 'fontname', 'times')
%     subplot(1,2,2);
%     plot(mean(md, 2), r, 'ko', 'markerfacecolor', [0 0 .8]);lsline
%     xlabel(sprintf('%s (tract num %d) MD', fgnames{fgnum}, fgnum),...
%         'fontsize', 14, 'fontname', 'times')
%     ylabel('Reading score','fontsize', 14, 'fontname', 'times')
%     
%     % Average tract profiles
%     h   = r>=120;
%     p   = r<=80;
%     g   = r>80 & r<120;
%     m1  = nanmean(fa(g==1, :));
%     m2  = nanmean(fa(p==1, :));
%     m3  = nanmean(fa(h==1, :));
%     
% 
%     % NOTE THAT WE SHOULD CALCULATE SE BASED ON THE NUMBER OF NON NAN SUBS
%     se1 = nanstd(fa(g==1, :))./sqrt(size(fa(g==1, :),1));
%     se2 = nanstd(fa(p==1, :))./sqrt(size(fa(p==1, :),1));
%     se3 = nanstd(fa(h==1, :))./sqrt(size(fa(h==1, :),1));
%     
% 
%     % plot them
%     figure; hold('on');
%     patch([1:100 fliplr(1:100)],[m1 - se1, fliplr(m1 + se1)],[.8 0 0],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
%     plot(1:100, m1, '-', 'color', [.8 0 0], 'linewidth',3);
%     patch([1:100 fliplr(1:100)],[m2 - se2, fliplr(m2 + se2)],[0 0 .8],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
%     plot(1:100, m2, '-', 'color', [0 0 .8], 'linewidth',3);
%     patch([1:100 fliplr(1:100)],[m3 - se3, fliplr(m3 + se3)],[0 .8 0],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
%     plot(1:100, m3, '-', 'color', [0 .8 0], 'linewidth',3);
% 
%     md1  = nanmean(md(g==1, :));
%     md2  = nanmean(md(p==1, :));
%     md3  = nanmean(md(h==1, :));
% 
%     semd1 = nanstd(md(g==1, :))./sqrt(size(md(g==1, :),1));
%     semd2 = nanstd(md(p==1, :))./sqrt(size(md(p==1, :),1));
%     semd3 = nanstd(md(h==1, :))./sqrt(size(md(h==1, :),1));
% 
%     figure; hold('on');
%     patch([1:100 fliplr(1:100)],[md1 - semd1, fliplr(md1 + semd1)],[.8 0 0],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
%     plot(1:100, md1, '-', 'color', [.8 0 0], 'linewidth',3);
%     patch([1:100 fliplr(1:100)],[md2 - semd2, fliplr(md2 + semd2)],[0 0 .8],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
%     plot(1:100, md2, '-', 'color', [0 0 .8], 'linewidth',3);
%     patch([1:100 fliplr(1:100)],[md3 - semd3, fliplr(md3 + semd3)],[0 .8 0],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
%     plot(1:100, md3, '-', 'color', [0 .8 0], 'linewidth',3);
end
