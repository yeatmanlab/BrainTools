% Function to load in and organize thickness data and sort it based on
% information derived from its respective Qdec file. 
% 
% Inputs:
% mgh_file: the path to the file produced by mri_surf2surf (after data was
% processed using mris_preproc)
% qdec_file: the path to the file containing relevant information about the data. 
%       **Important notes on this input**:
%           - It must be the qdec file used to produce the data in mgh_file
%           - The first column must be fsid, the second fsid-base, with all
%           following columns containing covariates. (If there are any, the
%           time variable should be the third column, as certain FreeSurfer
%           commands will default to that being the case. For more details,
%           see the documentation on qdec files in the FreeSurfer tutorial)
% 
% Outputs:
% vector containing (in order) the timepoint IDs (tpIDs), the design matrix (M), 
% data matrix (Y), the mri structure (mri), and vector containing the number 
% of repeated measures for each subject (ni)
% 
% Author: Mark Penrod
% Date: July 2017

function [tpIDs,M,Y,mri,ni] = long_prepare_LME(mgh_file,qdec_file)
%% Load in thickness data, spherical surface, and cortex labels
[Y,mri] = fs_read_Y(mgh_file);

%% Construct design matrix
Qdec = fReadQdec(qdec_file);

% save and then remove the column denoting time points
tpIDs = Qdec(2:end,1);
Qdec = rmQdecCol(Qdec,1);

% grab subject IDs (fsID-base) and then remove the column
sID = Qdec(2:end,1);

% convert the remaining data to a numeric matrix and sort the data
M = Qdec2num(Qdec);

% M = ordered design matrix, Y = ordered data matrix, and ni = a vector
% with the number of repeated measures for each subject
[M,Y,ni] = sortData(M,1,Y,sID);
end