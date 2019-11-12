    
 msh_path = '/mnt/scratch/PREK_Analysis/PreK_PD/nf_anatomy';
 cd(msh_path)
 voldata = 'TextvNontext.nii.gz';
 t1class = 't1_class.nii.gz';
 im = niftiRead(t1class);
 load msh_300.mat
 %fill_range = [0 1];
 fill_range = [2 4];
 [~,vRoi] = AFQ_meshDrawRoi(msh, [], voldata, fill_range,[], [],[]);
% [~,vRoi] = AFQ_meshDrawRoi(msh, [], voldata, [0 1],fill_range, [],[]);