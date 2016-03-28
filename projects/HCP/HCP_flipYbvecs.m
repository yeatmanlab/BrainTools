% Flip bvecs over Y axis


bv = dlmread('data.bvecs');
bv_yflip = bv;
bv_yflip(2,:) = bv_yflip(2,:).*-1;
dlmwrite('yflip.bvecs',bv_yflip)