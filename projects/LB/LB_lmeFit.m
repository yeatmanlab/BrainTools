%load /biac4/wandell/data/Lindamood_Bell/MRI/analysis/AFQ_sge_17-Feb-2014.mat
load ~/projects/Lindamood_Bell/MRI/analysis/AFQ_sge_17-Feb-2014.mat

property =  'md'
fgNames = AFQ_get(afq,'fgnames')
for ii = 1:20
   vals = AFQ_get(afq,fgNames{ii},property);
   % for now let's just use the intervention subjects
   vals = vals(1:24,:);
   sIds = afq.metadata.subIds(1:24)';
   time = repmat([-3 -1 1 3]',6,1);
   % fit linear mixed model
   tic
   for jj = 1:size(vals,2)
       d = dataset(vals(:,jj),sIds,time);
       d.sIds = nominal(d.sIds);
       lme= fitlme(d,'Var1 ~ time + (1|sIds)');
       pval(ii,jj) = lme.Coefficients.pValue(2);
   end
   toc
end