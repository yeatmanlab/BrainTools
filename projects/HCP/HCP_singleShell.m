function HCP_singleShell(dMRI, bvecs, bvals, brange, outname)
% Extract one shell of data from a multibvalue dataset
% 
% dtiExtractSingleShell(dMRI, bvecs, bvals, brange, outname)
im = readFileNifti(dMRI)
b = dlmread(bvals)
bv = dlmread(bvecs)
% if size(bv,2) > size(bv,1)
%     bv = bv';
% end
% if size(b,2) > size(b,1)
%     b = b';
% end
% % Scale small b values to 0
% b(b<=10)=0;

% Find dMRI volumes with bvalues in the specified range (or equal to zero)
v = (b > brange(1) & b < brange(2)) | b == 0;
% Extract image volumes and corresponding bvals
im.data = im.data(:,:,:,v); im.dim(4)=sum(v); im.fname = [outname '.nii.gz'];
b = b(v); bv = bv(v,:);
% Write out new nifti and corresponding bvals and bvecs
writeFileNifti(im);
dlmwrite([outname '.bval'],b');
dlmwrite([outname '.bvec'],bv');

