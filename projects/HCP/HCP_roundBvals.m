function HCP_roundBvals(bvalsFile, rbvals)
% Read in a .bvals file and round bvalues
%
% roundBvals(bvalsFile, rbvals)
%
% Example:
%
% bvalsFile = '/home/dstrodtman/Documents/HCP/Testing 100307/Raw/data.bvals'
% rbvals = [0 1000 2000 3000]
% roundBvals(bvalsFile, rbvals)

% Load in bvals text file
b = dlmread(bvalsFile);

% We're only dealing with the b=0 volumes for now
b(b<10) = 0;

% write the file out
dlmwrite(bvalsFile,b,'\t')