
% Read in bvecs file, flip bvecs over X axis


bv = dlmread('data.bvecs');
bv_xflip = bv;
bv_xflip(1,:) = bv_xflip(1,:).*-1;
dlmwrite('xflip.bvecs',bv_xflip)