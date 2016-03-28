function dwParams = HCP_params
%Define dwParams for HCP data processing

dwParams = dtiInitParams;
dwParams.clobber = 1; % overwrites previous data ***comment off after testing is complete
dwParams.eddyCorrect = -1; % turns off corrections done during pre-processing
dwParams.numBootStrapSamples = 1; % reduced for speed
dwParams.dwOutMm = [1.25 1.25 1.25]; % matches voxel size of data
dwParams.phaseEncodeDir = 2; % sets proper phase encode direction
