%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create high res head surface
%
% Usage: mkheadsurf -subjid
%
% by Sung Jun Joo 2017.02.27 @ University of Washington
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
sDir = '/mnt/diskArray/projects/avg_fsurfer';

cd(sDir)

d = dir('NLR*')

%%
origEnv = getenv('SUBJECTS_DIR')
setenv('SUBJECTS_DIR','/mnt/diskArray/projects/avg_fsurfer')
for iSubject = 1: length(d)
    system(sprintf('mkheadsurf -s %s', d(iSubject).name));
end
setenv('SUBJECTS_DIR',origEnv)