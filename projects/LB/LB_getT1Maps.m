function [path, im] = LB_getT1Maps

path = {'/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session1/LB4_20130807/20130807_1120/SPGR_1/Align_0.9375_0.9375_1.5/maps/T1_map_lsq.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session2/LB4_20130906/20130906_1538/SPGR_1/Align_0.9375_0.9375_1.5/maps/T1_map_lsq.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session3/LB4_20130927/20130927_1512/SPGR_1/Align_0.9375_0.9375_1.5/maps/T1_map_lsq.nii.gz'...
    '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/session4/LB4_20131120/20131120_1054/SPGR_1/Align_0.9375_0.9375_1.5/maps/T1_map_lsq.nii.gz'};

if nargout == 2
    for ii = 1:length(path)
        im(ii)=readFileNifti(path{ii});
    end
end

return

m = mean(cat(4,im(:).data),4);
% calculate difference from mean image
for ii = 1:length(im)
    im(ii).data(im(ii).data<0) = 0;
    im(ii).data(im(ii).data>4.5) = 4.5;
    
    d(:,:,:,ii) = (im(ii).data-m);
    showMontage(d(:,:,:,ii));colorbar;colormap jet;caxis([-.50 .50]);
end
c = reshape(corr

hist2d(im(1).data(:),im(2).data(:),.1:.1:4,.1:.1:4);
caxis([500 10000]);axis square