% script which will execute the make_correlation_map function for various
% covariates

[lh_tpID,lh_M,lh_Y,lh_mri,lh_ni] = ...
    long_prepare_LME('./thickness_preproc/lh.all_subs_tp1.thickness.stack.fwhm10.mgh','./qdec/all_subs_tp1_num.qdec.table.dat');
[rh_tpID,rh_M,rh_Y,rh_mri,rh_ni] = ...
    long_prepare_LME('./thickness_preproc/rh.all_subs_tp1.thickness.stack.fwhm10.mgh','./qdec/all_subs_tp1_num.qdec.table.dat');
lh_brs_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_brs_corr.mgh';
lh_twre_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_twre_corr.mgh';
lh_calc_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_calc_corr.mgh';
lh_mff_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/lh.tp1_allsubs_mff_corr.mgh';
rh_brs_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_brs_corr.mgh';
rh_twre_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_twre_corr.mgh';
rh_calc_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_calc_corr.mgh';
rh_mff_corr_outpath = '/mnt/scratch/projects/freesurfer/corr/rh.tp1_allsubs_mff_corr.mgh';

make_correlation_map(lh_Y,lh_mri,lh_M(:,2),lh_brs_corr_outpath);
make_correlation_map(rh_Y,rh_mri,rh_M(:,2),rh_brs_corr_outpath);
make_correlation_map(lh_Y,lh_mri,lh_M(:,3),lh_twre_corr_outpath);
make_correlation_map(rh_Y,rh_mri,rh_M(:,3),rh_twre_corr_outpath);
make_correlation_map(lh_Y,lh_mri,lh_M(:,4),lh_calc_corr_outpath);
make_correlation_map(rh_Y,rh_mri,rh_M(:,4),rh_calc_corr_outpath);
make_correlation_map(lh_Y,lh_mri,lh_M(:,5),lh_mff_corr_outpath);
make_correlation_map(rh_Y,rh_mri,rh_M(:,5),rh_mff_corr_outpath);

clust_path = '/mnt/scratch/projects/freesurfer/cluster_summaries';
clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_brs_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_brs_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_M(:,2),'BRS',lh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/BRS_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_brs_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_brs_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_M(:,2),'BRS',rh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/BRS_corr');

clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_twre_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_twre_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_M(:,3),'TWRE',lh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/TWRE_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_twre_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_twre_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_M(:,3),'TWRE',rh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/TWRE_corr');

clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_calc_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_calc_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_M(:,4),'CALC',lh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/CALC_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_calc_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_calc_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_M(:,4),'CALC',rh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/CALC_corr');

clust_sum = readtable(fullfile(clust_path,'lh.tp1_allsubs_mff_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'lh.tp1_allsubs_mff_corr.clusternum.mgh'),clust_sum ...
    ,'LH',lh_M(:,5),'MFF',lh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/MFF_corr');

clust_sum = readtable(fullfile(clust_path,'rh.tp1_allsubs_mff_corr.clustersum.xls'));
corrplot_ROIs(fullfile(clust_path,'rh.tp1_allsubs_mff_corr.clusternum.mgh'),clust_sum ...
    ,'RH',rh_M(:,5),'MFF',rh_Y,...
    '/mnt/scratch/projects/freesurfer/corr/MFF_corr');
