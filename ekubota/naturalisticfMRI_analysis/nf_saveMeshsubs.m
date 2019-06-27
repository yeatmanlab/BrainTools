function nf_saveMeshsubs(subList)


for ii = 1:length(subList)
    anat_dir = strcat('/mnt/scratch/PREK_Analysis/',subList{ii},'/ses-pre/t1');
    cd(anat_dir)

    t1class = 't1_class.nii.gz';
    im = niftiRead(t1class);
    msh = AFQ_meshCreate(t1class);
    save('msh.mat','msh')
end 
    
