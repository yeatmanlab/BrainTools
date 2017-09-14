%
% A function which will take information about the clusters identified as
% through some form of analysis and organize them into a matrix that can be
% used to plot the data.
%
% Inputs:
% clustnum: The numerical cluster file, the third output of mri_surfcluster
% or do_mri_surfcluster.m
% clust_sum: The cluster summary file
% tp_Ms: A cell array of the design matrices for each time point. The
% easiest way to get these is through the long_prepare_LME.m function.
% tp_Ys: A cell array of the data matrices for each time point. Again, the
% simplest way to produce these is through the long_prepare_LME.m function.
% time: a boolean which will determine what time variable is used. If time
% is listed as true then whatever variable is listed in column three of the
% design matrices will be used as the time variable (or rather listed as
% such). Otherwise the time variable used will be session number.
%
% Outputs:
% lme_long_mtx: A 2D matrix containing three columns:
%   (1) The subject IDs
%   (2) The cortical thickness at the time point and ROI
%   (3) Either a continous time variable or session number
%   (4) The name of the ROI, taken from the peak vertex of the cluster
%
% Author: Mark Penrod
% Date: August 4, 2017

function [lme_long_mtx,ROIs] = cluster_2_lme_longmtx(clustnum,clust_sum,tp_Ms,tp_Ys,time)
[lme_vol,lme_M] = load_mgh(clustnum);
rr = 1;
clusters = zeros(1,max(lme_vol));
for ii = 1:numel(lme_vol)
    if(lme_vol(ii) ~= 0)
        rr = 1;
        while rr <= rows(clusters)
            if clusters(rr,lme_vol(ii)) ~= 0
                rr = rr + 1;
            else
                break
            end
        end
        clusters(rr,lme_vol(ii)) = ii;
    end
end
lme_long_mtx = [];
for ii = 1:cols(clusters)
    % eliminate all trailing zeros from the cluster data
    vertices = clusters(find(clusters(:,ii),1,'first'):find(clusters(:,ii),1,'last'),ii);
    first_M = tp_Ms{1};
    first_Y = tp_Ys{1};
    for ll = 1:rows(first_M)
        ROI_means{ll,1} = mean(first_Y(ll,vertices));
    end
    ROIs{ii} = strcat(clust_sum.Annot{ii},'_',num2str(ii));
    ROI_names = cell(rows(first_M),1);
    ROI_names(:) = ROIs(ii);
    if time
        temp_mtx = [num2cell(first_M(:,1)),ROI_means,num2cell(first_M(:,3)),ROI_names];
    else
        temp_mtx = [num2cell(first_M(:,1)),ROI_means,num2cell(ones(rows(first_M),1)),ROI_names];
    end
    for jj = 2:numel(tp_Ms)
        clear ROI_means;
        clear ROI_names;
        curr_M = tp_Ms{jj};
        curr_Y = tp_Ys{jj};
        ROI_names = cell(rows(curr_M),1);
        ROI_names(:) = ROIs(ii);
        for ll = 1:rows(curr_M)
            ROI_means{ll,1} = mean(curr_Y(ll,vertices));
        end
        if time
            temp_mtx = [temp_mtx;num2cell(curr_M(:,1)),ROI_means,num2cell(curr_M(:,3)),ROI_names];
        else
            temp_mtx = [temp_mtx;num2cell(curr_M(:,1)),ROI_means,num2cell(ones(rows(curr_M),1)+(jj-1)),ROI_names];
        end
        clear ROI_names;
    end
    lme_long_mtx = [lme_long_mtx;temp_mtx];
end
end