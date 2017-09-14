function fs_base_recon(subID)
    freesurf_dir = '/mnt/scratch/projects/freesurfer/';
    for ii = 1:5
        if ~exist(fullfile(freesurf_dir, strcat(subID,'_', num2str(ii))), 'file')
            break
        end
    end
    num_sess = ii - 1;
    recon_cmd = strcat('recon-all -base', strcat([' ', subID], '_template'));
    for ii = 1:num_sess
        recon_cmd = strcat(recon_cmd, [' ','-tp'],...
            strcat([' ', subID], '_', num2str(ii)));
    end
    system(recon_cmd)
end