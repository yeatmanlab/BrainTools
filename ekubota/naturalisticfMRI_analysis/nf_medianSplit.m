%function nf_medianSplit

fsavgout = '/mnt/scratch/projects/freesurfer/fsaverage_maps/nf';
T = readtable('ses_pre.xlsx');

for ii = 1:height(T) 
    T.readingScore(ii) = sum([T.letknow_numlet(ii) T.letknow_numsound(ii)]);
end 

medianRS = median(T.readingScore);

low = T(T.readingScore<= medianRS,1);
high = T(T.readingScore>medianRS,1);
[~,~,exclude] = nf_excludeMotion(full_sublist);

[~,tmp] = strtok(exclude,'_');
[exclude,~] = strtok(tmp,'_');
exclude = str2double(exclude);

for ii = 1:length(exclude)
    low(low.record_id == exclude(ii),:) = [];
    high(high.record_id == exclude(ii),:) = [];
end 


hemis = {'lh','rh'};
contrasts = {'AllvBaseline.mgz','TextvNontext.mgz','reliabilityCorrelation.mgz'};

for h = 1:length(hemis)
    for c = 1:length(contrasts)
        for ii = 1:height(low)
            mapnames_low{ii} = sprintf('%s_PREK_%d_%s',hemis{h},low.record_id(ii),contrasts{c});
        end 
        for ii = 1:height(high)
            mapnames_high{ii} = sprintf('%s_PREK_%d_%s',hemis{h},high.record_id(ii),contrasts{c});
        end 
 
    
 
        for m = 1:length(mapnames_low)
            [con_low(:,m) M{m}] = load_mgh(fullfile(fsavgout,mapnames_low{m}));
        end 
    
        for m = 1:length(mapnames_high)
            [con_high(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mapnames_high{m}));
        end  
        
        [~,~,~,stats] = ttest2(con_low,con_high);
        T = stats.tstat';
        save_mgh(T,fullfile(fsavgout,sprintf('Tstat_MS_%s_%s',hemis{h},contrasts{c})),M{1});
    end 
end
 
    
% mapnames= {'lh_PREK_*_AllvBaseline.mgz','rh_PREK_*_AllvBaseline.mgz',...
%     'lh_PREK_*_reliabilityCorrelation.mgz', 'rh_PREK_*_reliabilityCorrelation.mgz',...
%     'lh_PREK_*_TextvNontext.mgz','rh_PREK_*_TextvNontext.mgz'};
% outnames= {'lh_PREK_AllvBaseline.mgz','rh_PREK_AllvBaseline.mgz',...
%     'lh_PREK_reliabilityCorrelation.mgz', 'rh_PREK_reliabilityCorrelation.mgz',...
%     'lh_PREK_TextvNontext.mgz','rh_PREK_TextvNontext.mgz'};
% 
% for m = 1:length(mapnames)
%     mgz = dir(fullfile(fsavgout,mapnames{m}));
%     
%     for ii = 1:length(mgz)
%         [con(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mgz(ii).name));
%     end
%     [~,~,~,stats] = ttest(con');
%     T = stats.tstat';
%     save_mgh(T,fullfile(fsavgout,sprintf('Tstat_%s',outnames{m})),M{1});
% end
