function nf_mvpa_timecourse(roiName,session)

full_sublist = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964','PREK_1887','PREK_1939',...
    'PREK_1505','PREK_1868','PREK_1208','PREK_1271','PREK_1372','PREK_1382',...
    'PREK_1673','PREK_1921','PREK_1936','PREK_1869','PREK_1443','PREK_1812',...
    'PREK_1714','PREK_1391','PREK_1293','PREK_1790','PREK_1878','PREK_1210',...
    'PREK_1706','PREK_1768','PREK_1401','PREK_1490','PREK_1818','PREK_1751',...
    'PREK_1103','PREK_1184', 'PREK_1798','PREK_1302','PREK_1460','PREK_1110','PREK_1756',...
    'PREK_1966','PREK_1750','PREK_1940','PREK_1262','PREK_1113','PREK_1241'};

[~,include,~] = nf_excludeMotion(full_sublist,session)

subs = include;
% 

%c = zeros(98,98,length(subs));
c = zeros(98,length(subs));
for si = 1:length(subs)
    betaPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si,1},'/',session,'/func');
    denoisedPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si,1},'/',session,'/func/GLMdenoise');
    anatPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si,1},'/',session,'/t1'); 
    roiPath = strcat('/mnt/disks/scratch/PREK_Analysis/data/',subs{si,1},'/fsROIs');
    
    % Load in inplane 
    cd(denoisedPath)
    %im1 = readFileNifti(fullfile(denoisedPath,['denoisedGLMrun0',int2str(subs{si,2}),'.nii']));
    im1 = readFileNifti(fullfile(betaPath,['run0',int2str(subs{si,2}),'.nii']));
    ref = mean(im1.data,4);
    ref_dimensions = im1.pixdim(1:3);
    
    cd(anatPath)
    
    % load in t1_acpc
    im3 = readFileNifti(fullfile(anatPath,'t1_acpc.nii.gz'));
    t1 = im3.data;
    t1_dimensions = im3.pixdim;

    % Load in alignment
    load(fullfile(anatPath,'tr.mat'))
    
    cd(roiPath)
    if exist(roiName,'file')
        im2 = readFileNifti(fullfile(roiPath,roiName));
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
        
        
        % Get matrix of time points (ROI size x number of predictors)
        for ii = 1:size(im1.data,4) %size(betas,4)
                bimage = squeeze(im1.data(:,:,:,ii));
                roidata1(:,ii) = bimage(roi_inplane == 1);
        end
        
         
        
         % convert to double and zscore
        roidata1 = zscore(double(roidata1)); 
        c(:,si) = mean(roidata1)';

%         % an mvpa like analysis
%         figure
%         c(:,:,si) = 1 - corr(roidata1);
%         imagesc(c(:,:,si)); colorbar;
    else
        c(:,si) = NaN;
        %c(:,:,si) = NaN;
    end
    clear roidata1 
end 
labels = readcell('/mnt/disks/scratch/PREK_Analysis/code/BrainTools/ekubota/naturalisticfMRI_analysis/postcamp_labels.csv')
X(:,1) = subs(:,1);
X(:,2) = labels;
for ii=1:size(c,2)
    X(ii,3) = {c(:,ii)};
end 
X = sortrows(X,2);
for ii=1:size(c,2)
    foo(:,ii) = X{ii,3}
end 
dist = corr(foo);
imagesc(dist);colorbar;
figure;hold on;
for ii = 1:size(labels,1)
    if strcmp(labels{ii},'letter')
        plot(1:98,c(:,ii),'Color',[1 0 0])
    else strcmp(labels{ii},'language')
        plot(1:98,c(:,ii),'Color',[0 0 1])
    end 
end 
% cd /mnt/disks/scratch/PREK_Analysis/code/BrainTools/ekubota/naturalisticfMRI_analysis/classification
% [~,roinameWoExt] = fileparts(roiName)
% [~,roinameWoExt] = fileparts(roinameWoExt)
% filename = strcat('/',roinameWoExt,'_',session);
% h5create('data.h5',filename,size(c))
% h5write('data.h5',filename,c)
%writematrix('c',filename)

% dm = zeros(length(subs),length(subs))
% for si = 1:length(subs)
%     for ss = 1:length(subs)
%         dm(si,ss) = abs(mean(mean(c(:,:,si) - c(:,:,ss))));
%     end 
% end 
% figure;imagesc(dm); colorbar;
% 
% dsm = squareform(pdist(dm,'correlation'));
% % multidimensional scaling 
% matrixemb = cmdscale(dsm,2)
% figure;
% scatter(matrixemb(:,1),matrixemb(:,2))
