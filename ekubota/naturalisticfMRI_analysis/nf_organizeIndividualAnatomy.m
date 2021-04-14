function nf_organizeIndividualAnatomy(sub_num)

root_dir = '/mnt/disks/scratch/PREK_Analysis/data/';
T1w = strcat('/mnt/disks/scratch/anatomy/',sub_num,'/t1_acpc.nii.gz');
ribbon = strcat('/mnt/disks/scratch/freesurferRecon/',sub_num,'/mri/ribbon.mgz');
anat_dir = strcat(root_dir, sub_num,'/ses-post/t1');
copyfile (T1w,anat_dir)
copyfile(ribbon, anat_dir)

ribbonfile = strcat(anat_dir,'/ribbon.mgz');
outfile = strcat(anat_dir,'/t1_class.nii.gz');
alignTo = strcat(anat_dir,'/t1_acpc.nii.gz');

cd(anat_dir)
fillWithCSF = true; 
fs_ribbon2itk(ribbonfile, outfile, fillWithCSF, alignTo)
