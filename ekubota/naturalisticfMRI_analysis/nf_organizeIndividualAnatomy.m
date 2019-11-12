function nf_organizeIndividualAnatomy(sub_num)

root_dir = '/mnt/scratch/PREK_Analysis/';
T1w = strcat('/mnt/diskArray56/projects/anatomy/',sub_num,'/t1_acpc.nii.gz');
ribbon = strcat('/mnt/diskArray56/projects/avg_fsurfer/',sub_num,'/mri/ribbon.mgz');
anat_dir = strcat(root_dir, sub_num,'/ses-pre/t1');
copyfile (T1w,anat_dir)
copyfile(ribbon, anat_dir)

ribbonfile = strcat(anat_dir,'/ribbon.mgz');
outfile = strcat(anat_dir,'/t1_class.nii.gz');
alignTo = strcat(anat_dir,'/t1_acpc.nii.gz');

cd(anat_dir)
fillWithCSF = true; 
fs_ribbon2itk(ribbonfile, outfile, fillWithCSF, alignTo)
