function fsl_preprocess(dwi_files, bvecs_file, bvals_file, pe_dir, outdir)
% Correct for EPI distortions, eddy currents and motion with FSL
%
% dwi_files  = Cell array of paths to niftis with alternating PE directions
% pe_dir     = mxn matrix denoting the phase encode direction of each
%              acquisition. Each row is for an image and columns are x,y,z
% bvecs_file = Bvecs file for each nifti image
% bvals_file = Bvals file for each nifti image
%
% example:
% dwi_files = ...
% {'/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84LRs051a001.nii.gz'...
% '/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84RLs049a001.nii.gz'};
% bvecs_file = ...
% {'/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84LRs051a001.bvec'...
% '/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84RLs049a001.bvec'};
% bvals_file = ...
% {'/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84LRs051a001.bval'...
% '/mnt/diskArray/projects/KNK/data/20140814S015/20140814_191546DWIdir84RLs049a001.bval'};
% pe_dir = [-1 0 0; 1 0 0];
% outdir = '/mnt/diskArray/projects/KNK/data/20140814S015/fsl_84dir'
% fsl_preprocess(dwi_files, bvecs_file, bvals_file, pe_dir, outdir)

%% Argument checking
if ~exist('outdir','var') || isempty(outdir)
    outdir = fileparts(dwi_files{1});
end
if ~exist(outdir,'dir')
    mkdir(outdir);
end
cd(outdir);

%% Concatenate files and pull out b=0 images
b0_cat = []; dfull = [];
for ii = 1:length(dwi_files)
    % load image
    im = readFileNifti(dwi_files{ii});
    % make a concetenated image
    dfull = cat(4,dfull,im.data);
    % load bvals and bvecs
    bvals{ii} = dlmread(bvals_file{ii});
    bvecs{ii} = dlmread(bvecs_file{ii});
    % define b0 volumes. Sometimes these still have minimal diffusion
    % weighting
    b0 = bvals{ii}<20;
    % pull out b0 volumes
    im.data = im.data(:,:,:,b0);
    im.dim(4) = sum(b0);
    %     b0name{ii} = [im.fname(1:end-7) '_b0s.nii.gz'];
    %     im.fname = b0name{ii};
    %     writeFileNifti(im);
    % make a concatenated b0
    b0_cat = cat(4,b0_cat,im.data);
    % mark which b0s in the concatenated image are from each acquisition
    nb0(ii) = sum(b0);
end
% Make a concatenated b0 image
im.data = b0_cat;
im.dim(4) = size(im.data,4);
b0cat_file = fullfile(outdir,'b0_cat.nii.gz');
im.fname = b0cat_file;
writeFileNifti(im);
% Write a text file with PE directions
pe = [];
for ii = 1:length(nb0)
    pe = vertcat(pe,repmat(pe_dir(ii,:),[nb0(ii),1]));
end
% The fourth column is the readout dwell time thing. This should be read
% from the images....
pe(1:end,4) = .0651;
acq_file=fullfile(outdir,'acqparams.txt');
dlmwrite(acq_file,pe,'\t');

% Write out a concatenated DWI
im.data = dfull;
im.dim(4) = size(im.data,4);
dwicat_file = fullfile(outdir,'dMRI_cat.nii.gz');
im.fname = dwicat_file;
writeFileNifti(im);
totalvols = im.dim(4);

% Write out concatenated bvecs and bvals
bvals_cat = horzcat(bvals{:});
% bvalues below 20 will be treated as 0
bvals_cat(bvals_cat<20) = 0;
bvecs_cat = horzcat(bvecs{:});
dlmwrite(fullfile(outdir,'bvecs_cat.bvec'),bvecs_cat,'\t');
dlmwrite(fullfile(outdir,'bvals_cat.bval'),bvals_cat,'\t');

%% run topup as a system call
cmd = sprintf('topup --imain=%s --datain=%s --config=b02b0.cnf --out=topup_results --iout=topup_b0',...
    b0cat_file,acq_file);
system(cmd);

%% brain extraction
system('fslmaths topup_b0 -Tmean topup_b0');
system('bet topup_b0 topup_b0_brain -m -f 0.2');

%% Create an index file
% First find the indices of the volumes within dwi_cat that are b0
in = find(bvals_cat==0); 
% Next we create an index where each dwi volume is associated with the b0
% that came before it. This way the topup params are applied to the nearest
% image in time. One other thing to keep in mind is that this reference
% should be relative to the concatenated b0 volume not the full dwi dataset
indx = [];
for ii = 1:totalvols
    % for volume ii find the it's difference in time from each b0 volume
    tdiff = in-ii;
    % Don't considder b0 collected later in time
    tdiff(tdiff>0) = inf;
    % Find the nearest b0 in time
    [~, i] = min(abs(tdiff));
    % Record the index to this volume
    % indx(ii) = in(i);
    indx(ii) = i;
end
dlmwrite('index.txt',indx,'\t');

%% run eddy currect correction
if ~exist('eddy','dir')
    mkdir('eddy');
end
outname = 'eddy/eddy_corrected_data';
cmd = sprintf('eddy --imain=dMRI_cat.nii.gz --mask=topup_b0_brain_mask.nii.gz --acqp=acqparams.txt --index=index.txt --bvecs=bvecs_cat.bvec --bvals=bvals_cat.bval --topup=topup_results --out=%s --flm=quadratic --niter=%d --verbose',...
    outname,8);
system(cmd);
% rotate bvecs - first load eddy paramters
b = dlmread([outname '.eddy_parameters']);
bvec = dlmread('bvecs_cat.bvec');
% rotate vecs. see: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq
bvecrot = [];
for ii = 1:size(bvec,2)
    bvecrot(:,ii) = fsl_rotMatrixFromEddy(b(ii,4),b(ii,5),b(ii,6))*bvec(:,ii);
end
dlmwrite('eddy/bvecs',bvecrot,'\t');
copyfile('bvals_cat.bval','eddy/bvals');
movefile([outname '.nii.gz'],'eddy/data.nii.gz')
copyfile('topup_b0_brain_mask.nii.gz', 'eddy/nodif_brain_mask.nii.gz')

%% Run dtifit
eddyNii   = fullfile(pwd,'eddy/data.nii.gz');
eddyBvecs = fullfile(pwd,'eddy/bvecs');
eddyBvals = fullfile(pwd,'eddy/bvals');
mask      = fullfile(pwd,'eddy/nodif_brain_mask.nii.gz');
dtidir    = fullfile(pwd,'dtifit');
dtiOut    = fullfile(dtidir,'dti');
if ~exist(dtidir,'dir')
    mkdir(dtidir);
end
cmd = sprintf('dtifit --data=%s --out=%s --mask=%s --bvecs=%s --bvals=%s',...
    eddyNii, dtiOut, mask, eddyBvecs, eddyBvals);
system(cmd);

%% Run bedpostx
%system('bedpostx eddy')
