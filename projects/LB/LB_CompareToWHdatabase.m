afq_lb=load('/biac4/wandell/data/Lindamood_Bell/MRI/analysis/AFQ_sge_17-Feb-2014.mat')
afq = load('/biac4/wandell/data/WH/analysis/WH_database_current.mat');
valname = 'rd';
fgNames = AFQ_get(afq.afq,'fgnames');
for ii = 1:20
    lb = AFQ_get(afq_lb.afq,fgNames{ii},valname);
    lb_m(ii) = nanmean(lb(:));
    lb_se(ii) = nanstd(nanmean(lb,2))./sqrt(6);
    wh = AFQ_get(afq.afq,fgNames{ii},valname);
    % remove subjects outside the age range
    wh = wh(afq.afq.metadata.age<=9 & afq.afq.metadata.age>=8,:);
    wh_m(ii) = nanmean(wh(:));
    wh_se(ii) = nanstd(nanmean(wh,2))./sqrt(size(wh,1));
    
end
figure; hold
errorbar(wh_m,wh_se,'ok');
errorbar(lb_m,lb_se,'or');
set(gca,'xticklabel',fgNames(1:20),'xtick',1:20)
rotateXLabels(gca,45)
ylabel(valname)
axis tight
legend('control','dyslexic')
%% Plot longitudinal
x = [0 1 2 60];
c = autumn(6);
sIds = unique(afq_lb.afq.metadata.subIds);
figure;
for ii = 1:20
     subplot(4,5,ii);hold;cnum=0;
    lb = AFQ_get(afq_lb.afq,fgNames{ii},valname);
    wh = AFQ_get(afq.afq,fgNames{ii},valname);
    % remove subjects outside the age range
    wh = wh(afq.afq.metadata.age<=9 & afq.afq.metadata.age>=8,:);
    wh_m(ii) = nanmean(wh(:));
    wh_sd(ii) = nanstd(nanmean(wh,2));
    %plot norms
    patch([x fliplr(x)],[repmat(wh_m(ii)-2.*wh_sd(ii),1,4) fliplr(repmat(wh_m(ii)+2.*wh_sd(ii),1,4))],[1 1 1])
    patch([x fliplr(x)],[repmat(wh_m(ii)-1.5.*wh_sd(ii),1,4) fliplr(repmat(wh_m(ii)+1.5.*wh_sd(ii),1,4))],[.8 .8 .8])
    patch([x fliplr(x)],[repmat(wh_m(ii)-wh_sd(ii),1,4) fliplr(repmat(wh_m(ii)+wh_sd(ii),1,4))],[.5 .5 .5])
    patch([x fliplr(x)],[repmat(wh_m(ii)-.5.*wh_sd(ii),1,4) fliplr(repmat(wh_m(ii)+.5.*wh_sd(ii),1,4))],[.3 .3 .3])
    plot(x,repmat(wh_m(ii),1,4),'-k','linewidth',2);
    s=0;
    for jj = 1:length(sIds)
        % find subjects with this id
        idx = afq_lb.afq.metadata.subIds == sIds(jj);
        % compute mean value
        y = nanmean(lb(idx,:),2);
        % calculate time
        t = cumsum([0 diff(afq_lb.afq.metadata.time(idx))]);
        t(t>60) = 60;
        if all(afq_lb.afq.sub_group(idx))
            cnum = cnum+1;
            plot(t,y,'-o','color',c(cnum,:),'markerfacecolor',c(cnum,:),'linewidth',2);
        else
            plot(t,y,'-o','color',[0 0 0],'markerfacecolor',[0 0 0],'linewidth',2);
        end
    end
    axis tight
    xlabel('days');
    ylabel(upper(valname));
    title(fgNames{ii});
end

%% Fit mixed models
valname =  'md'
fgNames = AFQ_get(afq_lb.afq,'fgnames')
for ii = 1:20
   vals = AFQ_get(afq_lb.afq,fgNames{ii},valname);
       wh = AFQ_get(afq.afq,fgNames{ii},valname);
    % remove subjects outside the age range
    wh = wh(afq.afq.metadata.age<=9 & afq.afq.metadata.age>=8,:);
    % Concat WH with LB data
    vals = vertcat(vals,wh);
   sIds = afq_lb.afq.metadata.subIds';
   % Add ids for WH subs
   sIds = vertcat(sIds,[101:100+size(wh,1)]');
   time = afq_lb.afq.metadata.session';
   % add in time points for the weston havens subjects
   time(end+1:end+size(wh,1)) = 0;
   % Variable for group
   group = zeros(size(time)); group(1:24) = 1;
   % fit linear mixed model
   tic
   % take the mean of vals
   %vals = nanmean(vals,2);
   for jj = 1:size(vals,2)
       d = dataset(vals(:,jj),sIds,time,group);
       d.sIds = categorical(d.sIds);
       d.group = categorical(d.group,[0 1],{'control' 'dyslexic'});
       lme= fitlme(d,'Var1 ~ time + group + (time|sIds)');
       pval(ii,jj,:) = lme.Coefficients.pValue(2:end);
       tstat(ii,jj,:) = lme.Coefficients.tStat(2:end);
   end
   toc
end

%% Fit mixed model just to LB subjects
valname =  'md'
fgNames = AFQ_get(afq_lb.afq,'fgnames')
for ii = 1:20
   vals = AFQ_get(afq_lb.afq,fgNames{ii},valname);
    vals = vals(1:24,:);
   
   sIds = afq_lb.afq.metadata.subIds(1:24)';
  
   time = afq_lb.afq.metadata.session(1:24)';


   % fit linear mixed model
   tic
   % take the mean of vals
   %vals = nanmean(vals,2);
   for jj = 1:size(vals,2)
       d = dataset(vals(:,jj),sIds,time);
       d.sIds = categorical(d.sIds);
       lme= fitlme(d,'Var1 ~ time + (1|sIds)');
       pval(ii,jj,:) = lme.Coefficients.pValue(2:end);
       tstat(ii,jj,:) = lme.Coefficients.tStat(2:end);
   end
   toc
end