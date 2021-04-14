subs = {'PREK_1112','PREK_1676','PREK_1691','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1372','PREK_1382',...
    'PREK_1673'};
roiName = '1007_ctx-lh-fusiform.nii.gz';
% 

c = zeros(3,3,length(subs));
for si = 1:length(subs)
    betaPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si},'/ses-pre/func');
    anatPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si},'/ses-pre/t1'); 
    roiPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si},'/fsROIs');
    
    % Get betas from results.mat
    cd(betaPath)
    load results.mat
    betas = results.modelmd{2};
    npredictors = size(betas,4)

    % Load in inplane 
    cd(betaPath)
    im1 = readFileNifti('run01.nii');
    ref = mean(im1.data,4);
    ref_dimensions = im1.pixdim(1:3);
    
    cd(anatPath)
    
    % load in t1_acpc
    im3 = readFileNifti('t1_acpc.nii.gz');
    t1 = im3.data;
    t1_dimensions = im3.pixdim;

    % Load in alignment
    load tr.mat   
    
    cd(roiPath)
    if exist(roiName,'file')
        im2 = readFileNifti(roiName);
        % load in ROI
        roi = im2.data;
        roi_dimensions = im2.pixdim;

        % Put ROI into inplane space.
        roi_inplane = extractslices(roi, roi_dimensions, ref(:,:,:,1), ref_dimensions, tr,0);
        
        % Threshold ROI image, make positive values 1, and turn Nans and
        % neg into zeros.
        roi_inplane(roi_inplane>0) = 1;
        roi_inplane(isnan(roi_inplane)) = 0;
        roi_inplane(roi_inplane<0) = 0;

        % Look at ROI on inplane
        % inplane(roi_inplane == 1) = max(inplane(:)) * 2;
        
        
        % Get matrix of beta weights (ROI size x number of predictors)
        for ii = 1:size(betas,4)
                bimage = squeeze(betas(:,:,:,ii));
                roidata(:,ii) = bimage(roi_inplane == 1);
        end
        
        
        mroidata = mean(roidata);

        % an mvpa like analysis
        figure
        c(:,:,si) = corr(roidata);
        imagesc(c(:,:,si));set(gca,'xtick',1:3,...
            'xticklabels', {'baseline' 'text' 'nontext'},...
            'ytick', 1:3,...
            'yticklabels', {'baseline' 'text' 'nontext'})
             caxis([.2 1]); colorbar;
    else
        c(:,:,si) = NaN;
    end
    clear roidata
end 

%cd /home/ekubota/git/BrainTools/ekubota/naturalisticfMRI_analysis
%filename = strcat(roiName,'_mvpa.mat')
%save(filename,'c')

