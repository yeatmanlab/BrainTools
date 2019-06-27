function nf_fsStats

fsavgout = '/mnt/scratch/projects/freesurfer/fsaverage_maps/nf';

%% Group level statistics
mapnames= {'lh_PREK_*_AllvBaseline.mgz','rh_PREK_*_AllvBaseline.mgz',...
    'lh_PREK_*_reliabilityCorrelation.mgz', 'rh_PREK_*_reliabilityCorrelation.mgz',...
    'lh_PREK_*_TextvNontext.mgz','rh_PREK_*_TextvNontext.mgz',...
    'lh_PREK_*_ModelFit.mgz','rh_PREK_*_ModelFit.mgz'};
outnames= {'lh_PREK_AllvBaseline.mgz','rh_PREK_AllvBaseline.mgz',...
    'lh_PREK_reliabilityCorrelation.mgz', 'rh_PREK_reliabilityCorrelation.mgz',...
    'lh_PREK_TextvNontext.mgz','rh_PREK_TextvNontext.mgz',...
    'lh_PREK_ModelFit.mgz','rh_PREK_ModelFit.mgz'};

for m = 1:length(mapnames)
    mgz = dir(fullfile(fsavgout,mapnames{m}));
    
    for ii = 1:length(mgz)
        [con(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mgz(ii).name));
    end
    [~,~,~,stats] = ttest(con');
    T = stats.tstat';
    save_mgh(T,fullfile(fsavgout,sprintf('Tstat_%s',outnames{m})),M{1});
end

%% average reliability maps. 
reliability = {'lh_PREK_*_reliabilityCorrelation.mgz',...
    'rh_PREK_*_reliabilityCorrelation.mgz'};
outnames = {'lh_PREK_reliabilityCorrelation.mgz', ...
    'rh_PREK_reliabilityCorrelation.mgz'};

for r = 1:length(reliability)
    mgz = dir(fullfile(fsavgout,reliability{r}));
    
    for ii = 1:length(mgz)
        [con(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mgz(ii).name));
    end 
    meanR = mean(con,2)
    save_mgh(meanR, fullfile(fsavgout,sprintf('avgR_%s',outnames{r})),M{1});
end 
            

modelfit = {'lh_PREK_*_ModelFit.mgz','rh_PRE_*_ModelFit.mgz'};
outnames = {'lh_PREK_ModelFit.mgz','rh_PREK_ModelFit.mgz'};

for r = 1:length(modelfit)
    mgz = dir(fullfile(fsavgout,modelfit{r}));
    
    for ii = 1:length(mgz)
        [con(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mgz(ii).name));
    end 
    rsquared = mean(con,2)
    meanR = sqrt(rsquared)
    save_mgh(rsquared, fullfile(fsavgout,sprintf('avgR2_%s',outnames{r})),M{1});
end 