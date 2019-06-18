function nf_organizeIndividualAnatomy(sub_num)

root_dir = '/mnt/scratch/PREK_Analysis/';
T1w = strcat('/mnt/diskArray56/projects/anatomy/',sub_num,'/t1_acpc.nii.gz');
ribbon = strcat('/mnt/diskArray56/projects/avg_fsurfer/',sub_num,'/mri/ribbon.mgz');
anat_dir = strcat(root_dir, sub_num,'/nf_anatomy');
copyfile (T1w,anat_dir)
copyfile(ribbon, anat_dir)