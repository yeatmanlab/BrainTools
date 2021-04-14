function nf_saveDenoisedasNifti(sessDir)

cd(sessDir)
mkdir('GLMdenoise')

im = readFileNifti(fullfile(sessDir,'run01.nii'));

load(fullfile('denoiseddata.mat'))

for i = 1:length(denoiseddata)
    im.fname = sprintf('denoisedGLMrun0%d.nii',i);
    im.data = denoiseddata{i};
    cd(fullfile(sessDir,'GLMdenoise'))
    writeFileNifti(im)
end 

