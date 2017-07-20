% function which constructs and runs the recon-all -base command based on a
% given subject ID
%
% Author: Mark Penrod
% Date: June 2017

function fs_base_recon(ID)
    freesurf_dir = '/mnt/scratch/projects/freesurfer/';
    addpath(genpath(freesurf_dir));
    for ii = 1:5
        if ~exist(fullfile(freesurf_dir, strcat(ID,'_', num2str(ii))), 'file')
            break
        end
    end
    num_sess = ii - 1;
    recon_cmd = strcat('recon-all -base', strcat([' ', ID], '_template'));
    for ii = 1:num_sess
        recon_cmd = strcat(recon_cmd, [' ','-tp'],...
            strcat([' ', ID], '_', num2str(ii)));
    end
    recon_cmd = [recon_cmd, ' -all'];
    system(recon_cmd)
end
