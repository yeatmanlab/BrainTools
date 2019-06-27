full_sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184'}

T = readtable('ses_pre.xlsx');

for ii = 1:height(T) 
    T.readingScore(ii) = sum([T.letknow_numlet(ii) T.letknow_numsound(ii)]);
end 

medianRS = nanmedian(T.readingScore);

low = T(T.readingScore< medianRS,1);
high = T(T.readingScore>=medianRS,1);
[~,~,exclude] = nf_excludeMotion(full_sublist);

[~,tmp] = strtok(exclude,'_');
[exclude,~] = strtok(tmp,'_');
exclude = str2double(exclude);

for ii = 1:length(exclude)
    low(low.record_id == exclude(ii),:) = [];
    high(high.record_id == exclude(ii),:) = [];
end 

%[C,include,exclude] = nf_excludeMotion(subs);
normedTimecourseLow = NaN(height(low),98);
timeCourseLow = NaN(height(low),98);
for ll = 1:height(low)
    timecoursePath = strcat('/mnt/scratch/PREK_Analysis/PREK_',int2str(low.record_id(ll)),'/ses-pre/t1');
    if exist(fullfile(timecoursePath,'timecourse_lfus.mat'),'file')
        load(fullfile(timecoursePath,'timecourse_lfus.mat'))
    end 
    timeCourseLow(ll,:) = squeeze(nanmean(timecourse_t1,1:3));
    normedTimecourseLow(ll,:) = zscore(timeCourseLow(ll,:));
end 

normedTimecourseHigh = NaN(height(high),98);
timeCourseHigh = NaN(height(high),98); 
for hh = 1:height(high)
    timecoursePath = strcat('/mnt/scratch/PREK_Analysis/PREK_',int2str(high.record_id(hh)),'/ses-pre/t1');
    if exist(fullfile(timecoursePath,'timecourse_lfus.mat'),'file')
        load(fullfile(timecoursePath,'timecourse_lfus.mat'))
    end
    timeCourseHigh(hh,:) = squeeze(nanmean(timecourse_t1,1:3));
    normedTimecourseHigh(hh,:) = zscore(timeCourseHigh(hh,:));    
end 

figure;hold on
stdshade(normedTimecourseLow,.4,'r');
stdshade(normedTimecourseHigh,.4,'b');
ylabel('zscore')
xlabel('time')
