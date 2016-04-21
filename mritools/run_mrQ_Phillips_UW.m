%% THIS IS A SCRIPT TO RUN IRTSE AND SPGR DATA THROUGH MRQ
% Get mrQ at: https://github.com/mezera/mrQ
% FOR PHILLIPS NIFTIS **MUST** BE CREATED USING 'fp' TO PROPERLY DEAL WITH
% PHILLIPS SCALING
% parrec2nii -c --scaling=fp *IRTSE*.PAR
% parrec2nii -c --scaling=fp *SPGR*.PAR

%% Data preparation
% create mrQ stricture and define the datadir where the nifti are  saved and outdir where mrQ output will be saved.
mrQ = mrQ_Create('/mnt/diskArray/projects/NLR/NLR_003/mrQ','mrQ_params_4_20');
% Check header
f = dir(fullfile(mrQ.RawDir ,'*.nii.gz'));
for ii = 1:length(f)
   impath = fullfile(mrQ.RawDir,f(ii).name);
   im = readFileNifti(impath);
   im = niftiCheckQto(im);
   writeFileNifti(im);
end
% Split real and magnitude images. The nifti array has 4 volumes that
% **shoould** be in the order Magnitude, Real, Imaginary, Phase. But please
% check
f = cat(1,dir(fullfile(mrQ.RawDir ,'*IRTSE*.nii.gz'),dir(fullfile(mrQ.RawDir ,'*SPGR*.nii.gz')));
for ii = 1:length(f)
    impath = fullfile(mrQ.RawDir,f(ii).name);
    im1 = readFileNifti(impath);
    im2 = im1;
    % volume 1 should be magnitude image
    im1.data = squeeze(im1.data(:,:,:,1));
    im1.dim(4) = 1;
    im1.fname = [im1.fname(1:end-7) '_mag.nii.gz']
    writeFileNifti(im1);
    if size(im2.data,4)>1
        im2.data = squeeze(im2.data(:,:,:,2));
        im2.dim(4) = 1;
        % volume 2 should be real
        im2.fname = [im2.fname(1:end-7) '_real.nii.gz'];
        writeFileNifti(im2);
    end

end
% Onecan set many different fit properties by mrQ_set.m
%
% Make a structure of  images and hdr info of the the nifti
% define the SEIR hdr info:
%
% mrQ.RawDir is the location where the  nifti are saved
inputData_spgr.rawDir = mrQ.RawDir;

% A list of nifti names  (a unique string from the names is enough)
% CHANGE THIS TO BE CORRECT IMAGE NAMES
inputData_spgr.name={'NLR_003_WIP_3D-SPGR-4_SENSE_12_1_mag.nii.gz'...
        'NLR_003_WIP_3D-SPGR-20_SENSE_11_1_mag.nii.gz'}

%
% the TR of each nifti in the list (msec)
inputData_spgr.TR=[14 14];
%
% The TE of each nifti in the list (msec)
inputData_spgr.TE=[2.3 2.3];
%
% The flip angle of each nifti in the list (degree)
inputData_spgr.flipAngle=[4 20];
%
% The  field strength of each nifti in the list (Tesla)
inputData_spgr.fieldStrength=[3 3];
%
% define the SEIR hdr info:
%
%   mrQ.RawDir is the location where the  nifti are saved
inputData_seir.rawDir=mrQ.RawDir;
%
% A list of nifti names  (a unique string from the names is enough)
% CHANGE THIS TO BE CORRECT IMAGE NAMES
inputData_seir.name={'NLR_003_WIP_IRTSE0050_opt_2mm_SENSE_7_1_mag.nii.gz'...
'NLR_003_WIP_IRTSE0480_opt_2mm_SENSE_8_1_mag.nii.gz'...
'NLR_003_WIP_IRTSE1200_opt_2mm_SENSE_9_1_mag.nii.gz'...
'NLR_003_WIP_IRTSE2400_opt_2mm_SENSE_10_1_mag.nii.gz'};

% the TR of each nifti in the list (msec)
inputData_seir.TR=[8525 8525 8525 8525];

% The TE of each nifti in the list (msec)
inputData_seir.TE=[49 49 49 49];

% The inversion time of each nifti in the list (msec)
inputData_seir.IT=[50  480 1200 2400];

% add the nifti info to the mrQ stracture
mrQ = mrQ_arrangeData_nimsfs(mrQ,inputData_spgr,inputData_seir);
%
%run it
mrQ=mrQ_Set(mrQ,'proclus',0)
mrQ=mrQ_Set(mrQ,'SunGrid',0)
save(mrQ.name,'mrQ')
mrQ_run(mrQ.name)
