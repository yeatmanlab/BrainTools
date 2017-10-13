
lh_BRS = lh_tp1_M(:,2);
rh_BRS = rh_tp1_M(:,2);

for ii = 1:rows(lh_V1_tp1.ThickAvg)
    lh_V1_tp1_CT(ii,1) = str2double(char(lh_V1_tp1.ThickAvg{ii}));
    lh_V2_tp1_CT(ii,1) = str2double(char(lh_V2_tp1.ThickAvg{ii}));
    rh_V1_tp1_CT(ii,1) = str2double(char(rh_V1_tp1.ThickAvg{ii}));
    rh_V2_tp1_CT(ii,1) = str2double(char(rh_V2_tp1.ThickAvg{ii}));
    lh_BA44_tp1_CT(ii,1) = str2double(char(lh_BA44_tp1.ThickAvg{ii}));
    lh_BA45_tp1_CT(ii,1) = str2double(char(lh_BA45_tp1.ThickAvg{ii}));
    rh_BA44_tp1_CT(ii,1) = str2double(char(rh_BA44_tp1.ThickAvg{ii}));
    rh_BA45_tp1_CT(ii,1) = str2double(char(rh_BA45_tp1.ThickAvg{ii}));
end
[lh_V1_tp1_corr,lh_V1_tp1_pval] = corr(lh_V1_tp1_CT(2:end,1),lh_BRS,'rows','pairwise');
[lh_V2_tp1_corr,lh_V2_tp1_pval] = corr(lh_V2_tp1_CT(2:end,1),lh_BRS,'rows','pairwise');
[rh_V1_tp1_corr,rh_V1_tp1_pval] = corr(rh_V1_tp1_CT(2:end,1),rh_BRS,'rows','pairwise');
[rh_V2_tp1_corr,rh_V2_tp1_pval] = corr(rh_V2_tp1_CT(2:end,1),rh_BRS,'rows','pairwise');
[lh_BA44_tp1_corr,lh_BA44_tp1_pval] = corr(lh_BA44_tp1_CT(2:end,1),lh_BRS,'rows','pairwise');
[lh_BA45_tp1_corr,lh_BA45_tp1_pval] = corr(lh_BA45_tp1_CT(2:end,1),lh_BRS,'rows','pairwise');
[rh_BA44_tp1_corr,rh_BA44_tp1_pval] = corr(rh_BA44_tp1_CT(2:end,1),rh_BRS,'rows','pairwise');
[rh_BA45_tp1_corr,rh_BA45_tp1_pval] = corr(rh_BA45_tp1_CT(2:end,1),rh_BRS,'rows','pairwise');
