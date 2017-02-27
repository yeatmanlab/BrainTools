% trans_acpc.m
% This function takes filelist as an input, ac-pc align, and average them.
% 
% Created by Sung Jun Joo and Libby Huber 2017.01.26 @ University of
% washington
% TODO: We need to automatize this.
%

%% AC-PC align
sub = 'NLR_132_WP'; % set the subject id

filelist = {'/mnt/diskArray/projects/MRI/NLR_132_WP/20160919/raw/NLR_132_WP_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_132_WP/20161010/raw/NLR_132_WP_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_132_WP/20161122/raw/NLR_132_WP_WIP_MEMP_VBM_SENSE_14_1.nii.gz'%, ...
%     '/mnt/diskArray/projects/MRI/NLR_132_WP/20160815/raw/NLR_102_RS_WIP_MEMP_VBM_SENSE_14_1.nii.gz'
    };

outpath = sprintf('/mnt/diskArray/projects/anatomy/%s',sub);

if isempty(dir(outpath))
    mkdir(outpath);
end
voxres = [0.8, 0.8, 0.8];

for i = 1: length(filelist)
    if isempty(dir(filelist{i}))
        system(sprintf('parrec2nii -o -c %s %s.par',filelist{i}(1:51),filelist{i}(1:end-7)));
    end
    temp = readFileNifti(filelist{i});
    tempmontage = makeMontage(temp.data,176:265);
    imagesc(tempmontage), colormap('gray')
end

% Only include good session numbers here!!!
gind = [1,3,4];

filelist = filelist(gind);

outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), [], voxres(1:3))