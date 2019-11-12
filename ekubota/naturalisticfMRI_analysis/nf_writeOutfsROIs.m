function nf_writeOutfsRois(subs)

for ii = 1:length(subs)
    fsIn = strcat('/mnt/diskArray56/projects/avg_fsurfer/',subs{ii},'/mri/aparc+aseg.mgz');
    type = 'nifti';
    outDir = strcat('/mnt/scratch/PREK_Analysis/',subs{ii},'/fsROIs');
    
    acpc_dir = strcat('/mnt/diskArray56/projects/anatomy/',subs{ii});
    if exist(fullfile(acpc_dir,'t1_acpc_avg.nii.gz'),'file')
        refT1 = strcat(acpc_dir,'/t1_acpc_avg.nii.gz');
    else
        refT1 = strcat(acpc_dir,'/t1_acpc.nii.gz');
    end 
    
    fs_roisFromAllLabels(fsIn,outDir,type,refT1);
end 

fsIn = strcat('/mnt/diskArray56/projects/avg_fsurfer/',subs{ii},'/mri/aparc+aseg.mgz');
    type = 'nifti';
    outDir = strcat('/mnt/scratch/PREK_Analysis/',subs{ii},'/fsROIs');
    
    acpc_dir = strcat('/mnt/diskArray56/projects/anatomy/',subs{ii});
    if exist(fullfile(acpc_dir,'t1_acpc_avg.nii.gz'),'file')
        refT1 = strcat(acpc_dir,'/t1_acpc_avg.nii.gz');
    else
        refT1 = strcat(acpc_dir,'/t1_acpc.nii.gz');
    end 
    
    fs_roisFromAllLabels(fsIn,outDir,type,refT1);