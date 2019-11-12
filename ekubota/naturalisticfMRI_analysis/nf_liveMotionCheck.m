

%% Preprocess data

data_dir = strcat('/home/ekubota/Desktop/motionCheck');
nf_preprocessfmri(data_dir)
cd(data_dir)
% Get functional filenames
EPIs = dir('run*.nii');

% Check how many functionals there are
nruns = size(EPIs);
nruns = nruns(1);

EPI_names = {};
% Create array of functional names
for ii = 1:nruns
    EPI_names = [EPI_names EPIs(ii).name];
end

%convert functionals, and set TR to 2.
convert_command = 'mri_convert --out_orientation RAS';

for ii = 1:nruns
    % Convert to RAS
    convert = [convert_command ' ' EPI_names{ii} ' ' EPI_names{ii}];
    system(convert);
    
    %Set TR to 2.2
    h = readFileNifti(EPI_names{ii});
    h.pixdim(4) = 2;
    h.time_units = 'sec';
    h.data = uint16(h.data);
    writeFileNifti(h);
end
