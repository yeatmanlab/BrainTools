% trans_acpc.m
% This function takes filelist as an input, ac-pc align, and average them.
% 
% Created by Sung Jun Joo and Libby Huber 2017.01.26 @ University of
% washington
% 
%

%% AC-PC align
sub = 'NLR_102_RS'; % set the subject id -- we're not using it at the moment...

% Change filelist per subject
% ATTN: If acpc aligned anatomy has been created, we should set the first
% filelist as the acpc aligned one in the 'anatomy' folder. And set the
% 'alignLandmarks' as false so the function below tries to align the
% subsequent images to the first one in the list assuming that is acpc
% aligned (which is true).
%
% Otherwise, just list mprage files in the MRI folder and set the
% 'alignLandmarks' as [] so the function below bring up the gui to acpc
% align and then average.
%
filelist = {'/mnt/diskArray/projects/anatomy/NLR_174_HS/t1_acpc.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_174_HS/20160722/raw/NLR_174_HS_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_174_HS/20160812/raw/NLR_174_HS_WIP_MEMP_VBM_SENSE_14_1.nii.gz', ...
    '/mnt/diskArray/projects/MRI/NLR_174_HS/20160829/raw/NLR_174_HS_WIP_MEMP_VBM_SENSE_16_1.nii.gz'
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
    if i == 1
        [PATHSTR,NAME,EXT] = fileparts(filelist{1})
        if ~strcmp(NAME,'t1_acpc.nii')
            temp = readFileNifti(filelist{i});
            tempmontage = makeMontage(temp.data,100:200);
            figure
            imagesc(tempmontage), colormap('gray')
            drawnow;
            keydown = waitforbuttonpress;
%             close(gcf);
        end
    else
        temp = readFileNifti(filelist{i});
        tempmontage = makeMontage(temp.data,150:250);
        figure
        imagesc(tempmontage), colormap('gray')
        drawnow;
        keydown = waitforbuttonpress;
%         close(gcf);
    end
end

%%
% UPDATE this so Only include good session numbers here!!!
gind = [1,2,3,4];

filelist = filelist(gind);

% outImg = mrAnatAverageAcpcNifti(fileNameList, outFileName, [alignLandmarks=[]], [newMmPerVox=[1 1 1]], [weights=ones(size(fileNameList))], [bb=[-90,90; -126,90; -72,108]'], [showFigs=true], [clipVals])
outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), false, voxres(1:3));

