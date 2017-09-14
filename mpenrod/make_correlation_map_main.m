% script which will execute the make_correlation_map function for various
% covariates
%% Load in data
[lh_tp1_tpID,lh_tp1_M,lh_tp1_Y,lh_tp1_mri,lh_tp1_ni] = ...
    long_prepare_LME('./thickness_preproc/lh.all_subs_tp1.thickness.stack.fwhm10.mgh','./qdec/all_subs_tp1_num.qdec.table.dat');
[rh_tp1_tpID,rh_tp1_M,rh_tp1_Y,rh_tp1_mri,rh_tp1_ni] = ...
    long_prepare_LME('./thickness_preproc/rh.all_subs_tp1.thickness.stack.fwhm10.mgh','./qdec/all_subs_tp1_num.qdec.table.dat');
[inter_tpID,inter_M,inter_lhY,inter_mri,inter_ni] = ...
    long_prepare_LME('./thickness_preproc/lh_intervention.thickness.stack.fwhm10.mgh','./qdec/inter_qdec/inter_long.qdec.table.dat');
[inter_tpID,inter_M,inter_rhY,inter_mri,inter_ni] = ...
    long_prepare_LME('./thickness_preproc/rh_intervention.thickness.stack.fwhm10.mgh','./qdec/inter_qdec/inter_long.qdec.table.dat');

lh_brs_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_brs_corr.mgh';
lh_twre_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_twre_corr.mgh';
lh_calc_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_calc_corr.mgh';
lh_mff_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_mff_corr.mgh';
lh_rf_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_rf_corr.mgh';
lh_age_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_age_corr.mgh';
rh_brs_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_brs_corr.mgh';
rh_twre_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_twre_corr.mgh';
rh_calc_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_calc_corr.mgh';
rh_mff_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_mff_corr.mgh';
rh_rf_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_rf_corr.mgh';
rh_age_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_age_corr.mgh';

%% Make Correlation Maps
% BRS
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,2),lh_brs_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,2),rh_brs_corr_outpath);
% TWRE
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,3),lh_twre_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,3),rh_twre_corr_outpath);
% Calc
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,4),lh_calc_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,4),rh_calc_corr_outpath);
% MFF
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,5),lh_mff_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,5),rh_mff_corr_outpath);
% RF
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,6),lh_rf_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,6),rh_rf_corr_outpath);
% Age
make_correlation_map(lh_tp1_Y,lh_tp1_mri,lh_tp1_M(:,7),lh_age_corr_outpath);
make_correlation_map(rh_tp1_Y,rh_tp1_mri,rh_tp1_M(:,7),rh_age_corr_outpath);

%% Make Correlation Plots
clust_path = '/mnt/scratch/projects/freesurfer/cluster_summaries';
% BRS
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_brs_corr.clustersum.xls'));
lh_brs_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_brs_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,2),'BRS',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/BRS_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_brs_corr.clustersum.xls'));
rh_brs_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_brs_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,2),'BRS',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/BRS_corr');

% TWRE
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_twre_corr.clustersum.xls'));
lh_twre_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_twre_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,3),'TWRE',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/TWRE_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_twre_corr.clustersum.xls'));
rh_twre_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_twre_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,3),'TWRE',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/TWRE_corr');

% CALC
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_calc_corr.clustersum.xls'));
lh_calc_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_calc_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,4),'CALC',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/CALC_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_calc_corr.clustersum.xls'));
rh_calc_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_calc_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,4),'CALC',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/CALC_corr');

% MFF
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_mff_corr.clustersum.xls'));
lh_mff_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_mff_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,5),'MFF',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/MFF_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_mff_corr.clustersum.xls'));
rh_mff_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_mff_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,5),'MFF',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/MFF_corr');

% RF
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_rf_corr.clustersum.xls'));
lh_rf_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_rf_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,6),'RF',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/RF_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_rf_corr.clustersum.xls'));
rh_rf_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_rf_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,6),'RF',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/RF_corr');

% Age
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_age_corr.clustersum.xls'));
lh_age_mtx = corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_age_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_tp1_M(:,7),'Age',lh_tp1_Y,inter_lhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/Age_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_age_corr.clustersum.xls'));
rh_age_mtx = corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_age_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_tp1_M(:,7),'Age',rh_tp1_Y,inter_rhY,[1,2,3,4],inter_tpID,...
    '/mnt/scratch/projects/freesurfer/corr/Age_corr');

%% Create longitudinal plots for all ROIs
longplot_dir = '/mnt/scratch/projects/freesurfer/long_plots';

brs_dir = fullfile(longplot_dir,'BRS_corr_long');
calc_dir = fullfile(longplot_dir,'CALC_corr_long');
mff_dir = fullfile(longplot_dir,'MFF_corr_long');
twre_dir = fullfile(longplot_dir,'TWRE_corr_long');
rf_dir = fullfile(longplot_dir,'RF_corr_long');
age_dir = fullfile(longplot_dir,'Age_corr_long');

% BRS
long_mtx_2_longplot(lh_brs_mtx,'LH','BRS',[1,2,3,4],brs_dir);
long_mtx_2_longplot(rh_brs_mtx,'RH','BRS',[1,2,3,4],brs_dir);
% Calc
long_mtx_2_longplot(lh_calc_mtx,'LH','Calc',[1,2,3,4],calc_dir);
long_mtx_2_longplot(rh_calc_mtx,'RH','Calc',[1,2,3,4],calc_dir);
% MFF
long_mtx_2_longplot(lh_mff_mtx,'LH','MFF',[1,2,3,4],mff_dir);
long_mtx_2_longplot(rh_mff_mtx,'RH','MFF',[1,2,3,4],mff_dir);
% TWRE
long_mtx_2_longplot(lh_twre_mtx,'LH','TWRE',[1,2,3,4],twre_dir);
long_mtx_2_longplot(rh_twre_mtx,'RH','TWRE',[1,2,3,4],twre_dir);
% RF
long_mtx_2_longplot(lh_rf_mtx,'LH','RF',[1,2,3,4],rf_dir);
long_mtx_2_longplot(rh_rf_mtx,'RH','RF',[1,2,3,4],rf_dir);
% Age
long_mtx_2_longplot(lh_age_mtx,'LH','Age',[1,2,3,4],age_dir);
long_mtx_2_longplot(rh_age_mtx,'RH','Age',[1,2,3,4],age_dir);

%% Consolidate the 3D matrices into 2D tables
% BRS
[lh_consol_brs_mtx,lh_brs_ROIs] = consol_long_mtx(lh_brs_mtx);
[rh_consol_brs_mtx,rh_brs_ROIs] = consol_long_mtx(rh_brs_mtx);
% TWRE
[lh_consol_twre_mtx,lh_twre_ROIs] = consol_long_mtx(lh_twre_mtx);
[rh_consol_twre_mtx,rh_twre_ROIs] = consol_long_mtx(rh_twre_mtx);
% MFF
[lh_consol_mff_mtx,lh_mff_ROIs] = consol_long_mtx(lh_mff_mtx);
[rh_consol_mff_mtx,rh_mff_ROIs] = consol_long_mtx(rh_mff_mtx);
% Calc
[lh_consol_calc_mtx,lh_calc_ROIs] = consol_long_mtx(lh_calc_mtx);
[rh_consol_calc_mtx,rh_calc_ROIs] = consol_long_mtx(rh_calc_mtx);
% RF
[lh_consol_rf_mtx,lh_rf_ROIs] = consol_long_mtx(lh_rf_mtx);
[rh_consol_rf_mtx,rh_rf_ROIs] = consol_long_mtx(rh_rf_mtx);
% Age
[lh_consol_age_mtx,lh_age_ROIs] = consol_long_mtx(lh_age_mtx);
[rh_consol_age_mtx,rh_age_ROIs] = consol_long_mtx(rh_age_mtx);

%% Fit to and plot out the lme longitudinal model
lme_long_dir = '/mnt/scratch/projects/freesurfer/long_plots/LME_longplots';

brs_dir = fullfile(lme_long_dir,'BRS');
twre_dir = fullfile(lme_long_dir,'TWRE');
mff_dir = fullfile(lme_long_dir,'MFF');
calc_dir = fullfile(lme_long_dir,'Calc');
age_dir = fullfile(lme_long_dir,'Age');

% BRS
lh_brs_lme = lme_long_fitandplot(lh_consol_brs_mtx,lh_brs_ROIs,'LH','BRS',brs_dir);
rh_brs_lme = lme_long_fitandplot(rh_consol_brs_mtx,rh_brs_ROIs,'RH','BRS',brs_dir);
% TWRE
lh_twre_lme = lme_long_fitandplot(lh_consol_twre_mtx,lh_twre_ROIs,'LH','TWRE',twre_dir);
rh_twre_lme = lme_long_fitandplot(rh_consol_twre_mtx,rh_twre_ROIs,'RH','TWRE',twre_dir);
% RF
lh_rf_lme = lme_long_fitandplot(lh_consol_rf_mtx,lh_rf_ROIs,'LH','RF',rf_dir);
rh_rf_lme = lme_long_fitandplot(rh_consol_rf_mtx,rh_rf_ROIs,'RH','RF',rf_dir);
% MFF
lh_mff_lme = lme_long_fitandplot(lh_consol_mff_mtx,lh_mff_ROIs,'LH','MFF',mff_dir);
rh_mff_lme = lme_long_fitandplot(rh_consol_mff_mtx,rh_mff_ROIs,'RH','MFF',mff_dir);
% Age
lh_age_lme = lme_long_fitandplot(lh_consol_age_mtx,lh_age_ROIs,'LH','Age',age_dir);
rh_age_lme = lme_long_fitandplot(rh_consol_age_mtx,rh_age_ROIs,'RH','Age',age_dir);
% Calc
lh_calc_lme = lme_long_fitandplot(lh_consol_calc_mtx,lh_calc_ROIs,'LH','Calc',calc_dir);
rh_calc_lme = lme_long_fitandplot(rh_consol_calc_mtx,rh_calc_ROIs,'RH','Calc',calc_dir);

% Freesurfer segmented ROIs
lh_inter_tbl = readtable('/mnt/scratch/projects/freesurfer/lh_intervention_long_aparc_stats.xls');
rh_inter_tbl = readtable('/mnt/scratch/projects/freesurfer/rh_intervention_long_aparc_stats.xls');

[lh_inter_consol_mtx,lh_inter_ROIs] = sumtable_2_consolmtx(lh_inter_tbl);
[rh_inter_consol_mtx,rh_inter_ROIs] = sumtable_2_consolmtx(rh_inter_tbl);

lh_inter_lme = lme_long_fitandplot(lh_inter_consol_mtx,lh_inter_ROIs,'LH','Intervention',fullfile(lme_long_dir,'Intervention'));
rh_inter_lme = lme_long_fitandplot(rh_inter_consol_mtx,rh_inter_ROIs,'RH','Intervention',fullfile(lme_long_dir,'Intervention'));
    
