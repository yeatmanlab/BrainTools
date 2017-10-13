% Function which calculates the correlations between the subjects cortical
% measurements and a given covariate and then creates a file which can be
% mapped onto the cortical surface.
%
% Inputs:
% data: the data for all subjects across all vertices (NxV matrix where N
% is the number of subjects and V is the number of vertices)
% mri: the mri structure (output of long_prepare_LME)
% covar: a vertex with the covariate, organized such that each covariate
% corresponds to the data at the same row in "data"
% outpath: the path to the folder + the desired file name to which the
% output will be saved
% 
% Author: Mark Penrod 
% Date: July 2017


function make_correlation_map(data,mri,covar,outpath)
corrs = zeros(1,cols(data));
for ii = 1:cols(data)
    corrs(ii) = corr(data(:,ii), covar(:,1), 'rows', 'pairwise');
end
mri.volsz = [163842 1];
fs_write_Y(corrs,mri,outpath);
end

