function outPath = mri_rms(inPath)
% Write out a mean squared combined image
%
% outPath = mri_rms(inPath)
%
% Inputs:
% inPath - Path to a nifti image
%
% Outputs:
% outPath - Path to the output image
%
% (c) University of Washington, Brain Development & Education Lab, 08/2015 

% Argument checking
if isempty(inPath) || ~exist(inPath,'file')
    error('Please provide a valid image path\n')
end

% Read in the image
im = readFileNifti(inPath);
% Combine teh four images into a mean squared image
im.data = sqrt(double(im.data(:,:,:,1).^2+im.data(:,:,:,2).^2+im.data(:,:,:,3).^2+im.data(:,:,:,4).^2));
% note in the header that there are no longer four images in dimension 4
im.dim(4) = 1;
% Change the image name to reflect the MS computation
im.fname = [prefix(prefix(inPath)) '_MSE.nii.gz'];
% Save the image back out
writeFileNifti(im);
% return the path to the image
outPath = im.fname;