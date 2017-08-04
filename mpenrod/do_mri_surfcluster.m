%
% Function that can be used in MATLAB to execute the mri_surfcluster
% command
%
% Inputs:
% mgh_file: name of the mgh file from which clusters will be extracted,
% this function should be modified to the accurate path to the mgh file
% hemi: lh or rh
% surface: the surface file to be used (ex. inflated)
% annot: the annotation file to be used (ex. aparc)
% min: the min value for cluster to be accepted
% max: the max value for which a cluster is accepted
% sign: pos, neg, or abs, which determines which sign of values are
% accepted for clusters
% outpath: path to folder where the cluster files will be saved
%
% Outputs:
% Three files:
%   (1) A summary file containing information on the peak vertex of the
%   cluster, its name, etc. This file should be edited into a file that can
%   be read in as a table to matlab, not including header information. The
%   data from this file must be read in that way to be compatible with
%   other functions
%   (2) A filtered cluster file in the same format as the original mgh
%   file, only instead of varying values all vertices have a 1 or a 0
%   depending on whether they are in a cluster or not
%   (3) A numerical clusters file in the same format as the original mgh
%   file, only instead of varying values within a cluster, each cluster has
%   its own value (which corresponds to the rows in the summary file). All
%   vertices in a given cluster will have this value.
%
% Author: Mark Penrod
% Date: August 2017

function do_mri_surfcluster(mgh_file,hemi,surface,annot,min,max,sign,outpath)
file_name = mgh_file(1:(strfind(mgh_file,'.mgh')-1));
cmd = strcat('mri_surfcluster --in /mnt/scratch/projects/freesurfer/',mgh_file,' --subject fsaverage --hemi',...
    [' ',hemi],' --surf',[' ',surface],' --annot', [' ',annot],' --thmin',...
    [' ',num2str(min)],' --thmax',[' ',num2str(max)],' --thsign',[' ',sign],...
    ' --sum',[' ',fullfile(outpath,strcat(file_name,'.clustersum.txt'))],...
    ' --o',[' ',fullfile(outpath,strcat(file_name,'.clusterfilter.mgh'))],...
    ' --ocn',[' ', fullfile(outpath,strcat(file_name,'.clusternum.mgh'))]);
 system(cmd)
end