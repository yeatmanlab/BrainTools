% Function which takes a the outputs of the mri_surfcluster function along
% with a covariate, calculates the correlation between the average
% thicknesses for the ROI and the covariate, and writes relevant
% information to a plot
%
% Inputs:
% clustnum_mgh: The cluster number surface output by mri_surfcluster
% clustsum: the summary file output by mri_surfcluster, read into matlab as
% a table
% hemi: the hemisphere being analyzed: 'LH' or 'RH'
% covar: a vertex containing the covariate
% covar_label: a string that is the name of the covariate
% data: the data output by long_prepare_LME
% outpath: path to folder which plots will be stored in 
%
% Author: Mark Penrod
% Date: July 19, 2017

function corrplot_ROIs(clustnum_mgh,clustsum,hemi,covar,covar_label,data,outpath)
% load in the cluster number surface
[vol,M] = load_mgh(clustnum_mgh);
clusters = zeros(1,max(vol));
% collect all clusters into a single matrix, where each column is a
% different cluster
for ii = 1:numel(vol)
    if(vol(ii) ~= 0)
        rr = 1;
        while rr <= rows(clusters)
            if clusters(rr,vol(ii)) ~= 0
                rr = rr + 1;
            else
                break
            end
        end
        clusters(rr,vol(ii)) = ii;
    end
end
for ii = 1:cols(clusters)
    vertices = [];
    % eliminate all trailing zeros from the cluster data
    vertices = clusters(find(clusters(:,ii),1,'first'):find(clusters(:,ii),1,'last'),ii);
    jj = 1;
    set(0,'DefaultFigureVisible','off');
    fig = figure;
    corr_val = 0;
    hold on
    consol_mtx = [];
    avg_thickness = [];
    % constructs a matrix where each column is the subjects data for each
    % vertex
    for rr = 1:numel(vertices)
        consol_mtx(:,jj) = data(:,vertices(rr));
        jj = jj + 1;
    end
    % calculate the average thickness across all vertices for each subject
    for rr = 1:rows(consol_mtx)
        avg_thickness(rr) = mean(consol_mtx(rr,:));
    end
    % plot the data a save
    plot(avg_thickness',covar,'.k','MarkerSize', 15)
    [corr_val,pval] = corr(avg_thickness',covar,'rows','pairwise');
    box off, lsline
    xlabel(strcat('Cortical Thickness in ',[' ', clustsum.Annot(ii)]))
    ylabel(covar_label)
    title(strcat('r= ', [' ',num2str(corr_val)],' pval=',[' ',num2str(pval)]))
    hold off
    saveas(fig,char(fullfile(outpath,strcat('corr_',hemi,clustsum.Annot(ii),'.jpg'))))
    set(0,'DefaultFigureVisible','on');
end