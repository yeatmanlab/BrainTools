% edit this to actual SUBJECTS_DIR
cd('/mnt/scratch/projects/freesurfer');

%% Load in thickness data, spherical surface, and cortex labels
[lhY,lh_mri] = fs_read_Y('./thickness_preproc/lh_intervention.thickness.stack.fwhm10.mgh');
lhsphere = fs_read_surf('fsaverage/surf/lh.sphere');
lhcortex = fs_read_label('fsaverage/label/lh.cortex.label');
[rhY,rh_mri] = fs_read_Y('./thickness_preproc/rh_intervention.thickness.stack.fwhm10.mgh');
rhsphere = fs_read_surf('fsaverage/surf/rh.sphere');
rhcortex = fs_read_label('fsaverage/label/rh.cortex.label');

%% Construct design matrix
Qdec = fReadQdec('inter_long.qdec.table.dat');
% remove the column denoting time points
Qdec = rmQdecCol(Qdec,1); 
% grab subject IDs (fsID-base) and then remove the column
sID = Qdec(2:end,1);
% remove age column
Qdec = rmQdecCol(Qdec,2);
% convert the remaining data to a numeric matrix and sort the data
M = Qdec2num(Qdec);
% M = ordered design matrix, Y = ordered data matrix, and ni = a vector 
% with the number of repeated measures for each subject
[M,lhY,ni] = sortData(M,1,lhY,sID);
[M,rhY,ni] = sortData(M,1,rhY,sID);

%% Parameter estimation and inference
% zcol = [ones(90,1) zeros(90,1)];
% [stats,st] = lme_mass_fit(M,[],[],zcol,Y,ni);

% [lh_stats,lh_st] = lme_mass_fit_vw(M,[1],lhY,ni,[],'lh_lme_stats',6);
% [rh_stats,rh_st] = lme_mass_fit_vw(M,[1],rhY,ni,[],'rh_lme_stats',6);

[lhTh0,lhRe] = lme_mass_fit_EMinit(M,[1],lhY,ni,lhcortex);
[rhTh0,rhRe] = lme_mass_fit_EMinit(M,[1],rhY,ni,rhcortex);

[lhRgs,lhRgMeans] = lme_mass_RgGrow(lhsphere,lhRe,lhTh0,lhcortex,2,95);
[rhRgs,rhRgMeans] = lme_mass_RgGrow(rhsphere,rhRe,rhTh0,rhcortex,2,95);

lhstats = lme_mass_fit_Rgw(M,[1],lhY,ni,lhTh0,lhRgs,lhsphere);
rhstats = lme_mass_fit_Rgw(M,[1],rhY,ni,rhTh0,rhRgs,rhsphere);

CM.C = [];
lh_fstats = lme_mass_F(lhstats,CM,[]);
rh_fstats = lme_mass_F(rhstats,CM,[]);


%% Correlation across subjects
% lhY and rhY matrices have each subject in a row, and each vertex in a
% column
% So, we want to calculate the correlation between each column and the BRS
% scores.

% look at the correlation across subjects for all time points
for ii = 1:size(lhY,2)
    brs_corr_lh(ii) = corr(lhY(:,ii), M(:,3), 'rows', 'pairwise');
    brs_corr_rh(ii) = corr(rhY(:,ii), M(:,3), 'rows', 'pairwise');
end

% look at the correlation across subjects for session 1
sess_ind = find(M(:,2) == 0);
for ii = 1:size(lhY,2)
    brs_corr_lh_s1(ii) = corr(lhY(sess_ind,ii), M(sess_ind,3), 'rows', 'pairwise');
    brs_corr_rh_s1(ii) = corr(rhY(sess_ind,ii), M(sess_ind,3), 'rows', 'pairwise');
end


vertex = 27629; %84494; %101373; %127986;
% plot correlation for session of interest
figure
plot(lhY(sess_ind,vertex), M(sess_ind,3), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel(strcat('Cortical Thickness at Vertex: ', num2str(vertex)))
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(brs_corr_lh_s1(vertex))))

% plot correlation across all sessions
figure
plot(lhY(:,vertex), M(:,3), '.k', 'MarkerSize', 15), hold on
box off, lsline
xlabel(strcat('Cortical Thickness at Vertex: ', num2str(vertex)))
ylabel('Basic Reading Score')
title(strcat('r= ', num2str(brs_corr_lh(vertex))))