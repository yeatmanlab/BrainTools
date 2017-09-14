% Function designed to take output of the mri_surfcluster function,
% analyze the clusters' correlation with a given covariate,
% and produce figures plotting significant findings
%
% Inputs:
% clust_sum_data: the relevant data extracted from a cleaned up version of
% the summary file (excluding all commented portions, leaving only the data
% table)
% data: data to be correlated with the covariate
% data_matrix: the data matrix 
% containing the subjects, and covariates of interest
%       ***NOTE***: the data_matrix must be organized in the same way as
%       the data with the exact same subjects such that all rows correspond
%       to the same subject between the two matrices
% covar_col: the column in the data_matrix which corresponds to the
% desired covariate for correlation
%
% Outputs:
% A matrix containg the vertex of max significance within the cluster (1),
% the name of the ROI in which that vertex lies (2), the correlations
% between that vertex and the covariate of interest (3), and the p-values
% of those correlations (4)
%
% Author: Mark Penrod
% Date: July 2017


function cluster_info_mtx = cluster_2_info_mtx(clust_sum_data,data,data_matrix,covar_col)

% collect and calculate the relevant data
for ii = 1:numel(clust_sum_data.VtxMax(:,1))
    cluster_info_mtx{ii,1} = clust_sum_data.VtxMax(ii,1);
    cluster_info_mtx{ii,2} = clust_sum_data.Annot{ii,1};
    [cluster_info_mtx{ii,3},cluster_info_mtx{ii,4}] = ...
        corr(data(:,clust_sum_data.VtxMax(ii,1)), data_matrix(:,covar_col), 'rows', 'pairwise');
end


% clean the matrix so only correlations with p < 0.05 and r > 0.3 remain
ii = 1;
endpoint = numel(cluster_info_mtx(:,1));
while ii <= endpoint
    if cluster_info_mtx{ii,4} > 0.05 || cluster_info_mtx{ii,3} < 0.3
        cluster_info_mtx(ii,:) = [];
        endpoint = endpoint - 1;
    else
        ii = ii + 1;
    end
end
end
