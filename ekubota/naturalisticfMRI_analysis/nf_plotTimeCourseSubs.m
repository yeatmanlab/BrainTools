full_sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184'};

[~,subList,~] = nf_excludeMotion(full_sublist);

roi_timecourse = nf_timecourseInFsRoi(subList);

T = readtable('ses_pre.xlsx');

for ii = 1:height(T) 
    T.readingScore(ii) = sum([T.letknow_numlet(ii) T.letknow_numsound(ii)]);
end 

medianRS = nanmedian(T.readingScore);

for ii = 1:height(T)
    if T.readingScore(ii) < medianRS 
        T.group(ii) = 1;
    elseif T.readingScore(ii) >= medianRS
        T.group(ii) = 2;
    end
end 

high_timecourse =[];
low_timecourse = [];
for ii = 1:length(subList)
    if T(strcmp(strcat('PREK_',int2str(T.record_id)),subList(ii)),29).group == 2
        high_timecourse = [high_timecourse; roi_timecourse(ii,:)];
    elseif T(strcmp(strcat('PREK_',int2str(T.record_id)),subList(ii)),29).group == 1
        low_timecourse = [low_timecourse; roi_timecourse(ii,:)];
    end
end 

for ll = 1:size(low_timecourse,1)
    normedTimecourseLow(ll,:) = zscore(low_timecourse(ll,:));
end 

for hh = 1:size(high_timecourse,1)
    normedTimecourseHigh(hh,:) = zscore(high_timecourse(hh,:));    
end 

figure;hold on
stdshade(normedTimecourseLow,.4,'r');
stdshade(normedTimecourseHigh,.4,'b');
ylabel('zscore')
xlabel('time')
