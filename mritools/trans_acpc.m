% trans_acpc.m
% This function takes filelist as an input, ac-pc align, and average them.
% 
% Created by Sung Jun Joo and Libby Huber 2017.01.26 @ University of
% washington
% TODO: We need to automatize this.
%

%% AC-PC align
filelist = {'/mnt/diskArray/projects/MRI/NLR_102_RS/20160618/raw/NLR_102_RS_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_102_RS/20160708/raw/NLR_102_RS_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_102_RS/20160726/raw/NLR_102_RS_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_102_RS/20160815/raw/NLR_102_RS_WIP_MEMP_VBM_SENSE_14_1.nii.gz'};
outpath = '/mnt/diskArray/projects/anatomy/test/';
if isempty(dir(outpath))
    mkdir(outpath);
end
voxres = [0.8, 0.8, 0.8];

system(sprintf('freeview -v %s',filelist{1}));

system(sprintf('freeview -v %s',filelist{2}));

system(sprintf('freeview -v %s',filelist{3}));

system(sprintf('freeview -v %s',filelist{4}));

gind = [1,3,4];

filelist = filelist(gind)

outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), [], voxres(1:3))