function dirList = HCP_autoDir(baseDir)
% Creates cell vector of all subject directories to allow automatization of
% pre-processing for HCP data.
%
% basedir should point to the directory that contains all of the HCP
% patient directories (folders with 6 digit subject ID for names).
%
% example:
% basedir = '/home/dstrodtman/Documents/HCP'
% dirList = HCP_autoDir(basedir)

% Read basedir
curDir = dir(baseDir);

% Find directories within basedir
subDirs = find(vertcat(curDir.isdir));

% Create dirList, seed with blank cells
dirList = cell(1, numel(subDirs)-2);

% Create index for feeding dirList
kk = 0;

% Index names of folders into dirList (first two entries are '.' '..', so
% skipped)
for ii = 3:numel(subDirs)
    kk = kk + 1;
    jj = subDirs(ii);
    dirList{kk} = curDir(jj).name;
end