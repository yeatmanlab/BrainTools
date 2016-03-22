sub_dirs = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB1_20130630/20130630_1437/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB1_20130716/20130716_1606/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB1_20130730/20130730_1004/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB1_20130818/20130818_1602/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB2_20130628/20130628_1812/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB2_20130715/20130715_1805/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB2_20130729/20130729_1738/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB2_20130808/20130808_1746/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB4_20130807/20130807_1120/raw/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB4_20130906/20130906_1538/raw/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB4_20130927/20130927_1512/raw/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB4_20131120/20131120_1054/raw/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB11_20130709/20130709_1008/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB11_20130731/20130731_1805/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB11_20130819/20130819_1012/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB11_20130909/20130909_1617/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB17_20130728/20130728_1345/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB17_20130813/20130813_1910/dti70trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB17_20130827/20130827_1835/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB17_20130910/20130910_1907/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB18_20130805/20130805_1025/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB18_20130820/20130820_1438/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB18_20130904/20130904_1540/dti111trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB18_20131016/20131016_1443/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB7_20130712/20130712_1717/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB7_20130802/20130802_1028/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB8_20130723/20130723_1304/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB8_20130813/20130813_1214/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB9_20130711/20130711_1616/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB9_20130807/20130807_1547/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB10_20130703/20130703_1148/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB10_20130717/20130717_1334/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB12_20130717/20130717_1743/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB12_20130806/20130806_1315/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB15_20130729/20130729_1325/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session2/LB15_20131017/20131017_1635/dti80trilin'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/control/session1/LB16_20130722/20130722_1744/dti80trilin'};
sub_group = [ones(1,24) zeros(1,13)];
afq = AFQ_Create('sub_dirs',sub_dirs,'sub_group',sub_group,'clip2rois',0,'outdir','/biac4/wandell/data/Lindamood_Bell/MRI/analysis')
%afq = AFQ_set(afq,'overwritefibers')

for ii = 1:length(afq.sub_dirs)
    % Find date
    idx = strfind(sub_dirs{ii},'2013');
    d = str2num(afq.sub_dirs{ii}(idx(1):idx(1)+7));
    afq.metadata.time(ii) = datenum(str2num(sub_dirs{ii}(idx(1):idx(1)+3)),str2num(sub_dirs{ii}(idx(1)+4:idx(1)+5)),str2num(sub_dirs{ii}(idx(1)+6:idx(1)+7)));
end
afq.metadata.subIds =  [1 1 1 1 2 2 2 2 4 4 4 4 11 11 11 11 17 17 17 17 18 18 18 18 7 7 8 8 9 9 10 10 12 12 15 15 16];
afq.metadata.session = [-3 -1 1 3 -3 -1 1 3 -3 -1 1 3 -3  -1  1  3  -3  -1  1  3  -3  -1  1  3  -1 1 -1 1 -1 1 -1  1  -1  1  -1  1  0];

%% Set up pzths to maps
for ii = 1:length(sub_dirs)
    r1map{ii} = fullfile(sub_dirs{ii},'bin', 'R1_map_lsq_2DTI.nii.gz')
    mtvmap{ii} = fullfile(sub_dirs{ii},'bin', 'TV_map_2DTI.nii.gz')
end
afq = AFQ_set(afq,'images',r1map);
afq = AFQ_set(afq,'images',mtvmap);

%% Run afq
afq = AFQ_run([],[],afq);
%afq = AFQ_run_sge(afq);

return
%% Compute some stats
%subnums = unique(afq.metadata.subIds);
subnums_lb = [1 2 4 11 17 18];
subnums_c = [7 8 9 10 12 15];
fgNames = AFQ_get(afq,'fgnames');
nodes = 45:51;
col = jet(length(subnums_lb));
figure
d=nan(6,4,20);
valname = 'md'
rn = ceil(rand(1,2).*100);
for jj = 1:20
    
    fa = AFQ_get(afq,fgNames{jj},valname);
    fa = nanmean(fa(:,nodes),2);
    for ii = 1:length(subnums_lb)
        d(ii,1:sum(afq.metadata.subIds == subnums_lb(ii)),jj) = fa(afq.metadata.subIds == subnums_lb(ii));
    end
    for ii = 1:length(subnums_c)
        d_c(ii,1:sum(afq.metadata.subIds == subnums_c(ii)),jj) = fa(afq.metadata.subIds == subnums_c(ii));
    end
    figure(rn(1));subplot(5,4,jj);hold;plot(d(:,:,jj)','-o');
    plot(d(:,:,jj)');
    %axis tight
    xlabel('Session');ylabel(valname);title(fgNames{jj})
    set(gca,'xtick',[1 2 3 4])
    
    d2(:,:,jj) = bsxfun(@minus,d(:,:,jj),nanmean(d(:,:,jj),2))+nanmean(flatten(d(:,:,jj)));
    d2_c(:,:,jj) = bsxfun(@minus,d_c(:,:,jj),nanmean(d_c(:,:,jj),2))+nanmean(flatten(d_c(:,:,jj)));
    
    m = nanmean(d2(:,:,jj));
    s = nanstd(d2(:,:,jj))./sqrt(6);
    m_c = nanmean(d2_c(:,:,jj));
    s_c = nanstd(d2_c(:,:,jj))./sqrt(6);
    
    figure(rn(2));subplot(5,4,jj);hold;
    errorbar(m,s,'-r')
    errorbar(m_c,s_c,'-k')
    
    %plot(d(:,:,jj)');
    %axis tight
    xlabel('Session');ylabel(valname);title(fgNames{jj})
    set(gca,'xtick',[1 2 3 4])
end
%%
% normalize d



[~,pval] = ttest(c);
figure;

for ii = 1:20
    subplot(5,4,ii);
    title(fgNames{ii})
    plot(norminv(pval(ii*100-99:ii*100)))
    axis([1 100 -2.5 2.5])
    grid on
    title(fgNames{ii})
    
end
%% Make some plots
fgNames = AFQ_get(afq,'fgnames');
colors = [1 .5 .5; 1 .5 .5; .5 0 0;.5 0 0; .5 .5 1; .5 .5 1; 0 0 .5;0 0 0.5];
nodes = 30:70;
for ii = 1:length(fgNames)
    figure; hold
    fa = AFQ_get(afq,fgNames{ii},'md');
    for jj = 1:size(fa,1)
        p(jj)=plot(nodes,fa(jj,nodes),'-','color',colors(jj,:),'linewidth',2);
    end
    title(fgNames{ii});
    legend(p(1:2:end),{'S1 early' 'S1 late' 'S2 early' 'S2 late'});
end
figure
for ii = 1:length(fgNames)
    subplot(4,5,ii)
    hold
    fa = AFQ_get(afq,fgNames{ii},'fa');
    fa = nanmean(fa(:,nodes),2);
    plot(1:4,fa(1:4),'-ro')
    plot(1:4,fa(5:8),'-bo')
    
    title(fgNames{ii});
end