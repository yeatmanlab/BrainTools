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
subList = {'NLR_205_AC'};

%%
origEnv = getenv('SUBJECTS_DIR')
setenv('SUBJECTS_DIR','/mnt/diskArray/projects/avg_fsurfer')
% for iSubject = 1: length(d)
%     system(sprintf('sudo chmod 777 %s/%s/scripts/mkheadsurf.log','/mnt/diskArray/projects/avg_fsurfer',d(iSubject).name));
% end

for iSubject = 1: length(subList)
%     system(sprintf('mkheadsurf -s %s', d(iSubject).name));
    system(sprintf('mkheadsurf -s %s', subList{iSubject}));
end
setenv('SUBJECTS_DIR',origEnv)