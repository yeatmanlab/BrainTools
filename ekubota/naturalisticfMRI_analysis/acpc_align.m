function acpc_align(sub_num)
% Takes in a path to a T1W image, requires user to set AC and PC and
% outputs an ACPC aligned image
cd /mnt/diskArray56/projects/anatomy
mkdir(sub_num)

anatPath = strcat('/mnt/scratch/PREK_Analysis/', sub_num,'/ses-pre/raw/');
cd(anatPath)
parCommand = strcat('parrec2nii -c -b',{' '},sub_num,'_1_WIP_MPRAGE_SENSE_6_1.PAR');
system(parCommand{1})

T1path = strcat('/mnt/scratch/PREK_Analysis/', sub_num,'/ses-pre/raw/', sub_num, '_1_WIP_MPRAGE_SENSE_6_1.nii.gz')

% T1path = mri_rms(T1path); % Root mean squared image
im = niftiRead(T1path); % Read root mean squared image
voxres =im.pixdim(1:3); % Get the voxel resolution of the image (mm)
mrAnatAverageAcpcNifti({T1path}, strcat('/mnt/diskArray56/projects/anatomy/', ...
    sub_num, '/t1_acpc.nii.gz'), [], voxres(1:3), [], [], [], [.01 .99])