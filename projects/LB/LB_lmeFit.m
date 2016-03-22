% Script to fit a linear mixed effects model (lme) to analyze longitudinal
% data from AFQ

% Load AFQ structure
%load /biac4/wandell/data/Lindamood_Bell/MRI/analysis/AFQ_sge_17-Feb-2014.mat
load ~/projects/Lindamood_Bell/MRI/analysis/AFQ_sge_17-Feb-2014.mat

% property to analyze
property =  'md'

% Get fiber group names
fgNames = AFQ_get(afq,'fgnames')

% Loop over fiber groups
for ii = 1:20
    % Get the values of interest for this fiber group
   vals = AFQ_get(afq,fgNames{ii},property);
   % for now let's just use the data from the intervention subjects
   vals = vals(1:24,:);
   % Subject ids noting which measurements came from which subjects
   sIds = afq.metadata.subIds(1:24)';
   % This is our effect of interest. We want to see changes over time. It
   % would probably be better to have this variable be hours of
   % intervention since timing varies across subjects
   time = repmat([-3 -1 1 3]',6,1);
   % fit linear mixed model
   tic % time this...it's slow
   for jj = 1:size(vals,2)
       % Make a matlab data frame
       d = dataset(vals(:,jj),sIds,time);
       % Subject ids must be a categorical variable
       d.sIds = nominal(d.sIds);
       % Fit the model with subjects as a random effect
       lme= fitlme(d,'Var1 ~ time + (1|sIds)');
       % Get the pvalues for the effect of interest. Did diffusion
       % properties change over time?
       pval(ii,jj) = lme.Coefficients.pValue(2);
   end
   toc
end