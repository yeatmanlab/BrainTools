sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};
dates = {'20190525','20190525','20190525','20190525','20190525',...
    '20190524','20190524','20190525','20190525'};

for ii = 9:length(sublist)
    refPath = strcat('/mnt/scratch/PREK_Analysis/',sublist{ii},'/',dates{ii},'/fmri');
    targetPath = strcat('/mnt/scratch/PREK_Analysis/',sublist{ii},'/nf_anatomy');
    alignmentPath = strcat('/mnt/scratch/PREK_Analysis/',sublist{ii},'/',dates{ii},'/fmri');
    
    % Load in inplane
    cd(refPath)
    im1 = readFileNifti('run01.nii');
    ref = mean(im1.data,4);
    ref_dimensions = im1.pixdim(1:3);

    % Load in t1
    cd(targetPath)
    if exist(fullfile(targetPath,'t1_acpc.nii.gz'),'file')
        im2 = readFileNifti(fullfile(targetPath,'t1_acpc.nii.gz'));
    else 
        im2 = readFileNifti(fullfile(targetPath,'t1_acpc_avg.nii.gz'));
    end 
    
    target = im2.data;
    target_dimensions = im2.pixdim;

    % get initial seed value
    alignvolumedata(target,target_dimensions,ref,ref_dimensions);
    keypress=0;
    fh = gcf;
    fprintf('\nPress any key in window %d to exit when you are happy with alignment\n',fh.Number)
    while keypress==0
        keypress=waitforbuttonpress;
    end
    tr = alignvolumedata_exporttransformation;
    
    % automate alignment with the seed
    
    alignvolumedata(target,target_dimensions,ref,ref_dimensions,tr);

    useMI = true;  % you need MI if the two volumes have different tissue contrast.
                   % it's much faster to not use MI.
    alignvolumedata_auto([],[],0,[4 4 2],[],[],[],useMI);  % rigid bo
   
    alignvolumedata_auto([],[],0,[1 1 1],[],[],[],useMI);  % rigid body, fine, mutual information metric
    
    % check alignment 
    keypress=0;
    fh = gcf;
    fprintf('\nPress any key in window %d to exit when you are happy with alignment\n',fh.Number)
    while keypress==0
        keypress=waitforbuttonpress;
    end    

    tr = alignvolumedata_exporttransformation;
    cd(targetPath)
    save('tr.mat','tr')
    
    

    close all
end