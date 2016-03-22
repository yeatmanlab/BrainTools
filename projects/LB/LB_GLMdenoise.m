fmridir = '/home/jyeatman/projects/Lindamood_Bell/MRI/child/intervention/all/LB18/fMRI/wordloc/'
tr = 2
for ii = 1:3
design{ii} = makeDesignMatrixFromParfile(fullfile(fmridir,'Stimuli/Parfiles','1_Loc_WordFaceObj.par'),tr,8);
end
stimdur = 16;
hrf = getcanonicalhrf(stimdur,tr);
rawdir = fullfile(fmridir,'RAW');
fmris = matchfiles(fullfile(rawdir, 'run*.nii*'));
for ii = 1:length(fmris)
   tmp = readFileNifti(fmris{ii});
   data{ii} = tmp.data(:,:,:,1:size(design{1},1));
end
[results, denoiseddata] = GLMdenoisedata(design, data, stimdur, tr, 'assume', hrf,[],'GLMdenoisefigures');