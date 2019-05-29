D = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};

for ii = 1:length(D)
    anat_dir = strcat('/mnt/scratch/PREK_Analysis/',D{ii},'/nf_anatomy');
    ribbonfile = strcat(anat_dir,'/ribbon.mgz');
    outfile = strcat(anat_dir,'/t1_class.nii.gz');
    alignTo = strcat(anat_dir,'/t1_acpc.nii.gz');

    cd(anat_dir)
    fillWithCSF = true; 
    fs_ribbon2itk(ribbonfile, outfile, fillWithCSF, alignTo)

    t1class = 't1_class.nii.gz';
    im = niftiRead(t1class);
    msh = AFQ_meshCreate(t1class);
    save('msh.mat','msh')
end 
    
