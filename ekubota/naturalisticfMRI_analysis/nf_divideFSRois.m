function nf_divideFSRois(subs)

for si = 1:length(subs)
    roiDir = strcat('/mnt/scratch/PREK_Analysis/',subs{si},'/fsROIs');
    cd(roiDir)

    roiNames = {'1007_ctx-lh-fusiform.nii.gz','1009_ctx-lh-inferiortemporal.nii.gz',...
    '1016_ctx-lh-parahippocampal.nii.gz', '2007_ctx-rh-fusiform.nii.gz',...
    '2009_ctx-rh-inferiortemporal.nii.gz','2016_ctx-rh-parahippocampal.nii.gz'};

    outNames = {'lh_fus','lh_itg','lh_ph','rh_fus','rh_itg','rh_ph'};

    for ii = 1:length(roiNames)
        im = readFileNifti(roiNames{ii});

        IND =find(im.data==1);

        s = [181,217,181];
        [I,J,K] = ind2sub(s,IND);

        range = max(J) - min(J);

        quarter = floor(range/4);

        q1 = [min(J) (min(J) + quarter)];
        q2 = [(min(J) + quarter + 1) (min(J) + 2*quarter +1)];
        q3 = [(min(J) + 2*quarter +2) (min(J) + 3*quarter+ 2)];
        q4 = [(min(J) + 3*quarter+3) max(J)];

        % write out most posterior roi;
        tmp = im.data;
        tmp(:,q1(2):217,:) = 0;

        im.data = tmp;
        im.fname = strcat(outNames{ii},'1.nii.gz');
        writeFileNifti(im);

        % write out next posterior roi
        im = readFileNifti(roiNames{ii});
        tmp = im.data;
        tmp(:,1:q2(1),:) = 0;
        tmp(:,q2(2):217,:) = 0;

        im.data = tmp;
        im.fname = strcat(outNames{ii},'2.nii.gz');
        writeFileNifti(im)

        % write out second-most anterior roi
        im = readFileNifti(roiNames{ii});
        tmp = im.data;
        tmp(:,1:q3(1),:) = 0;
        tmp(:,q3(2):217,:) = 0;

        im.data = tmp;
        im.fname = strcat(outNames{ii},'3.nii.gz');
        writeFileNifti(im)

        % write out most anterior roi
        im = readFileNifti(roiNames{ii});
        tmp = im.data;
        tmp(:,1:q4(1),:) = 0;
        tmp(:,q4(2):217,:) = 0;

        im.data = tmp;
        im.fname = strcat(outNames{ii},'4.nii.gz');
        writeFileNifti(im)
    end 
end 
