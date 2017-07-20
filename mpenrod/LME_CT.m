% edit this to actual SUBJECTS_DIR
cd('/mnt/scratch/projects/freesurfer');

%% Load in thickness data, spherical surface, and cortex labels and construct design matrix
% Cortex labels and spherical surface
lhsphere = fs_read_surf('fsaverage/surf/lh.sphere');
lhcortex = fs_read_label('fsaverage/label/lh.cortex.label');

rhsphere = fs_read_surf('fsaverage/surf/rh.sphere');
rhcortex = fs_read_label('fsaverage/label/rh.cortex.label');

% Intervention data and design matrix
[inter_tpID,inter_M,inter_lhY,inter_mri,inter_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_intervention.thickness.stack.fwhm10.mgh','./qdec/inter_long.qdec.table.dat');
[inter_tpID,inter_M,inter_rhY,inter_mri,inter_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_intervention.thickness.stack.fwhm10.mgh','./qdec/inter_long.qdec.table.dat');

% Control data and design matrix, excluding dyslexic controls
[control_exRI_tpID,control_exRI_M,control_exRI_lhY,control_exRI_mri,control_exRI_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_control_exRI.thickness.stack.fwhm10.mgh','./qdec/control_long_exRI_num.qdec.table.dat');
[control_exRI_tpID,control_exRI_M,control_exRI_rhY,control_exRI_mri,control_exRI_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_control_exRI.thickness.stack.fwhm10.mgh','./qdec/control_long_exRI_num.qdec.table.dat');

% Controls data and design matrix for all controls
[control_tpID,control_M,control_lhY,control_mri,control_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_control.thickness.stack.fwhm10.mgh','./qdec/control_long_num.qdec.table.dat');
[control_tpID,control_M,control_rhY,control_mri,control_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_control.thickness.stack.fwhm10.mgh','./qdec/control_long_num.qdec.table.dat');

% Data and design matrix for subjects with BRS score *over* 85 at timepoint 1
% through all timepoints
[highBRSatTP1_tpID,highBRSatTP1_M,highBRSatTP1_lhY,highBRSatTP1_mri,highBRSatTP1_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_highBRS_atTP1.thickness.stack.fwhm10.mgh','./qdec/highBRS_atTP1_num.long.qdec.table.dat');
[highBRSatTP1_tpID,highBRSatTP1_M,highBRSatTP1_rhY,highBRSatTP1_mri,highBRSatTP1_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_highBRS_atTP1.thickness.stack.fwhm10.mgh','./qdec/highBRS_atTP1_num.long.qdec.table.dat');

% Data and design matrix for subjects with BRS score *under* 85 at timepoint 1
% through all timepoints
[lowBRSatTP1_tpID,lowBRSatTP1_M,lowBRSatTP1_lhY,lowBRSatTP1_mri,lowBRSatTP1_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_lowBRS_atTP1.thickness.stack.fwhm10.mgh','./qdec/lowBRS_atTP1_num.long.qdec.table.dat');
[lowBRSatTP1_tpID,lowBRSatTP1_M,lowBRSatTP1_rhY,lowBRSatTP1_mri,lowBRSatTP1_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_lowBRS_atTP1.thickness.stack.fwhm10.mgh','./qdec/lowBRS_atTP1_num.long.qdec.table.dat');



%% Parameter estimation and inference
% zcol = [ones(90,1) zeros(90,1)];
% [stats,st] = lme_mass_fit(M,[],[],zcol,Y,ni);

% [lh_stats,lh_st] = lme_mass_fit_vw(M,[1],lhY,ni,[],'lh_lme_stats',6);
% [rh_stats,rh_st] = lme_mass_fit_vw(M,[1],rhY,ni,[],'rh_lme_stats',6);
for ii  = 1:rows(inter_M)
    if ii == 1 || inter_M(ii,1) ~= inter_M(ii-1,1)
        inter_M(ii,5) = 1;
    elseif inter_M(ii-1,5) == 1
        inter_M(ii,6) = 1;
    elseif inter_M(ii-1,6) == 1
        inter_M(ii,7) = 1;
    elseif inter_M(ii-1,7) == 1
        inter_M(ii,8) =1;
    end
end
inter_M = [inter_M(:,1),inter_M(:,5:end)];
[inter_lhTh0,inter_lhRe] = lme_mass_fit_EMinit(inter_M,[1],inter_lhY,inter_ni,lhcortex,[],6);
[inter_rhTh0,inter_rhRe] = lme_mass_fit_EMinit(inter_M,[1],inter_rhY,inter_ni,rhcortex,[],6);

[inter_lhRgs,inter_lhRgMeans] = lme_mass_RgGrow(lhsphere,inter_lhRe,inter_lhTh0,lhcortex,2,95);
[inter_rhRgs,inter_rhRgMeans] = lme_mass_RgGrow(rhsphere,inter_rhRe,inter_rhTh0,rhcortex,2,95);

inter_lhstats = lme_mass_fit_Rgw(inter_M,[1],inter_lhY,inter_ni,inter_lhTh0,inter_lhRgs,lhsphere);
inter_rhstats = lme_mass_fit_Rgw(inter_M,[1],inter_rhY,inter_ni,inter_rhTh0,inter_rhRgs,rhsphere);


CM.C = [0,-1,1,0,0;0,0,-1,1,0;0,0,0,-1,1];
inter_lh_fstats = lme_mass_F(inter_lhstats,CM,6);
inter_rh_fstats = lme_mass_F(inter_rhstats,CM,6);

% [lh_detvtx,lh_sided_pval,lh_pth] = lme_mass_FDR2(inter_lh_fstats.pval,inter_lh_fstats.sgn,lhcortex,0.05,0);
% [rh_detvtx,rh_sided_pval,rh_pth] = lme_mass_FDR2(inter_rh_fstats.pval,inter_rh_fstats.sgn,rhcortex,0.05,0);
% 
% fs_write_Y(lh_sided_pval,inter_mri,'lh_test_pval.mgh');
% fs_write_Y(rh_sided_pval,inter_mri,'rh_test_pval.mgh');

fs_write_fstats(inter_lh_fstats,inter_mri,'lh_test_sig.mgh','sig');
fs_write_fstats(inter_rh_fstats,inter_mri,'rh_test_sig.mgh','sig');

%% Correlation across subjects
% inter_lhY and inter_rhY matrices have each subject in a row, and each vertex in a
% column
% So, we want to calculate the correlation between each column and the BRS
% scores. IN THE INTERVENTION GROUP

% look at the correlation across subjects for all time points
for ii = 1:size(inter_lhY,2)
    inter_brs_corr_lh(ii) = corr(inter_lhY(:,ii), inter_M(:,3), 'rows', 'pairwise');
    inter_brs_corr_rh(ii) = corr(inter_rhY(:,ii), inter_M(:,3), 'rows', 'pairwise');
end

% look at the correlation across subjects for session 1
sess_ind = find(inter_M(:,3) == 0);
for ii = 1:size(inter_lhY,2)
    brs_corr_lh_s1(ii) = corr(inter_lhY(sess_ind,ii), inter_M(sess_ind,4), 'rows', 'pairwise');
    brs_corr_rh_s1(ii) = corr(inter_rhY(sess_ind,ii), inter_M(sess_ind,4), 'rows', 'pairwise');
end


vertex = 162277; %84494; %101373; %127986;
% plot correlation for session of interest
figure
plot(inter_lhY(sess_ind,vertex), inter_M(sess_ind,4), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel(strcat('Cortical Thickness at Vertex: ', num2str(vertex)))
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(brs_corr_lh_s1(vertex))))

% plot correlation across all sessions
figure
plot(inter_lhY(:,vertex), inter_M(:,3), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel(strcat('Cortical Thickness at Vertex: ', num2str(vertex)))
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(inter_brs_corr_lh(vertex))))

%% Analyzing regions of interest for correlations of various measures with cortical thickness in said regions

% Load and sort data
cd('/mnt/scratch/projects/freesurfer')
inter_lhstats = readtable('lh_intervention_long_aparc_stats.xls');
inter_rhstats = readtable('rh_intervention_long_aparc_stats.xls');
inter_lhstats = sortrows(inter_lhstats);
inter_rhstats = sortrows(inter_rhstats);

% Array with regions of interest
lhROIs = {'lhstats.lh_fusiform_thickness','lhstats.lh_lateraloccipital_thickness',...
    'lhstats.lh_inferiortemporal_thickness','lhstats.lh_lingual_thickness',...
    'lhstats.lh_MeanThickness_thickness','lhstats.lh_parahippocampal_thickness',...
    'lhstats.lh_superiortemporal_thickness','lhstats.lh_supramarginal_thickness'};
rhROIs = {'rhstats.lh_fusiform_thickness','rhstats.lh_lateraloccipital_thickness',...
    'rhstats.lh_inferiortemporal_thickness','rhstats.lh_lingual_thickness',...
    'rhstats.lh_MeanThickness_thickness','rhstats.lh_parahippocampal_thickness',...
    'rhstats.lh_superiortemporal_thickness','rhstats.lh_supramarginal_thickness'};

% look at the correlation across subjects for all time points, focusing on
% the [insert brain area here]
[inter_brs_corr_lh(1),brs_pval_lh(1)] = corr(inter_lhstats.lh_lingual_thickness(sess_ind,1), inter_M(sess_ind,3), 'rows', 'pairwise');
inter_brs_corr_rh(1) = corr(inter_rhstats.rh_supramarginal_thickness(:,1), inter_M(:,3), 'rows', 'pairwise');

figure
plot(inter_lhstats.lh_supramarginal_thickness(:,1), inter_M(:,3), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel('Cortical Thickness at LH supramarginal')
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(inter_brs_corr_lh(1))))

figure
plot(inter_rhstats.rh_supramarginal_thickness(:,1), inter_M(:,3), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel('Cortical Thickness at RH supramarginal')
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(inter_brs_corr_rh(1))))
%% Correlating BRS to vertices identified from clusters of interest
% Load data
lhclusters = readtable('/mnt/scratch/projects/freesurfer/cluster_summaries/brs_followtp1_1.lh.clustersum.xls');
rhclusters = readtable('/mnt/scratch/projects/freesurfer/cluster_summaries/brs_followtp1_1.rh.clustersum.xls');

% collect vertex of max (1), ROI name (2), correlations (3), and p-values
% (4)  into matrices
brs_cluster_corr_lh = cluster_2_info_mtx(lhclusters,inter_lhY,inter_M,4);
brs_cluster_corr_rh = cluster_2_info_mtx(rhclusters,inter_rhY,inter_M,4);

% plot out all the resulting significant vertices, relating cortical
% thickness at a given vertex to basic reading score
% plot correlation across all sessions
set(0,'DefaultFigureVisible','off');
for ii = 1:numel(brs_cluster_corr_lh(:,1))
    vertex = brs_cluster_corr_lh{ii,1};
    ROI = brs_cluster_corr_lh{ii,2};
    r = brs_cluster_corr_lh{ii,3};
    pval = brs_cluster_corr_lh{ii,4};
    figure
    myplot = plot(inter_lhY(:,vertex), inter_M(:,3), '.k', 'MarkerSize', 15); hold on
    box off, lsline
    xlabel(strcat('Cortical Thickness at in LH', [' ',ROI],' at Vertex',[' ',num2str(vertex)]))
    ylabel('Basic Reading Score')
    title(strcat('r=', [' ',num2str(r)],' p-val=',[' ',pval]))
    hold off 
    saveas(myplot,strcat('/mnt/scratch/projects/freesurfer/corr/LH/',ROI,'_',num2str(vertex),'_brs_corr.jpg'));
end

for ii = 1:numel(brs_cluster_corr_rh(:,1))
    vertex = brs_cluster_corr_rh{ii,1};
    ROI = brs_cluster_corr_rh{ii,2};
    r = brs_cluster_corr_rh{ii,3};
    pval = brs_cluster_corr_rh{ii,4};
    figure
    myplot = plot(inter_rhY(:,vertex), inter_M(:,3), '.k', 'MarkerSize', 15); hold on
    box off, lsline
    xlabel(strcat('Cortical Thickness at in RH', [' ',ROI],' at Vertex',[' ',num2str(vertex)]))
    ylabel('Basic Reading Score')
    title(strcat('r=', [' ',num2str(r)],' p-val=',[' ',pval]))
    hold off 
    saveas(myplot,strcat('/mnt/scratch/projects/freesurfer/corr/RH/',ROI,'_',num2str(vertex),'_brs_corr.jpg'));
end
set(0,'DefaultFigureVisible','on');

%% Creating plots to show the change in cortical thickness across timepoints comparing control group to intervention group
outpath = '/mnt/scratch/projects/freesurfer/long_plots/highBRStp1_v_lowBRStp1_long';
cluster_2_longplot(highBRSatTP1_tpID,lowBRSatTP1_tpID,'LH',brs_cluster_corr_lh(:,1),...
    brs_cluster_corr_lh(:,2),highBRSatTP1_lhY,lowBRSatTP1_lhY,outpath);
cluster_2_longplot(highBRSatTP1_tpID,lowBRSatTP1_tpID,'RH',brs_cluster_corr_rh(:,1),...
    brs_cluster_corr_rh(:,2),highBRSatTP1_rhY,lowBRSatTP1_rhY,outpath);



