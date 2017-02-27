% trans_acpc.m
% This function takes filelist as an input, ac-pc align, and average them.
% 
% Created by Sung Jun Joo and Libby Huber 2017.01.26 @ University of
% washington
% TODO: We need to automatize this.
%

%% AC-PC align
sub = 'NLR_187_NB'; % set the subject id -- we're not using it at the moment...

% Change filelist per subject
filelist = {'/mnt/diskArray/projects/MRI/NLR_187_NB/20161017/raw/NLR_187_NB_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_187_NB/20161103/raw/NLR_187_NB_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_187_NB/20161119/raw/NLR_187_NB_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_187_NB/20161205/raw/NLR_187_NB_WIP_MEMP_VBM_SENSE_14_1.nii.gz'
    };
 
outpath = sprintf('/mnt/diskArray/projects/anatomy/%s',sub);

if isempty(dir(outpath))
    mkdir(outpath);
end
voxres = [0.8, 0.8, 0.8];

for i = 1: length(filelist)
    if isempty(dir(filelist{i}))
        system(sprintf('parrec2nii -c -o %s %s.par',filelist{i}(1:51),filelist{i}(1:end-7)));
    end
    temp = readFileNifti(filelist{i});
    tempmontage = makeMontage(temp.data,176:265);
    imagesc(tempmontage), colormap('gray')
    drawnow;
    pause;
end

% Only include good session numbers here!!!
gind = [1,2,3];

filelist = filelist(gind);

outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), [], voxres(1:3));