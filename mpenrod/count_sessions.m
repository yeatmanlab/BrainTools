function num_sess = count_sessions(ID)
    freesurf_dir = '/mnt/scratch/projects/freesurfer/';
    addpath(genpath(freesurf_dir));
    long_cmds = cell(5);
    for ii = 1:5
        if ~exist(fullfile(freesurf_dir, strcat(ID,'_', num2str(ii))), 'file')
            break
        end
    end
    num_sess = ii - 1;
end

