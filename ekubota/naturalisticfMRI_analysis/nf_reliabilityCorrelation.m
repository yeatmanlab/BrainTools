function R = nf_reliabilityCorrelation(subject)

functionalPath = strcat('/mnt/scratch/PREK_Analysis/',subject,'/ses-pre/func/GLMdenoise');

cd(functionalPath)

im1 = readFileNifti('denoisedGLMrun01.nii');
im2 = readFileNifti('denoisedGLMrun02.nii');


R = zeros(64,64,33);
for ii = 1:64 
    for jj = 1:64
        for kk = 1:33 
            R(ii,jj,kk) = corr(double(squeeze(im1.data(ii,jj,kk,:))),double(squeeze(im2.data(ii,jj,kk,:))));
        end
    end 
end 

showMontage(R)
colormap jet