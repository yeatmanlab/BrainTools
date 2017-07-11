% function which constructs and runs the recon-all -long command based on a
% given subject ID
function long_cmds = fs_long_recon(ID)
    freesurf_dir = '/mnt/scratch/projects/freesurfer/';
    long_cmds = cell(5);
    num_sess = 1;
    for ii = 1:5
        % whether there is a session to process
        if ~exist(fullfile(freesurf_dir, strcat(ID,'_', num2str(ii))), 'file') 
            break
        % if it's already been processed
        elseif exist(fullfile(freesurf_dir, strcat(ID,'_',num2str(ii),...
                '.long.',ID,'_template')),'dir')
            continue
        else
            long_cmds{num_sess} = strcat('recon-all -long', strcat([' ',ID],'_',num2str(ii)),...
            strcat([' ',ID],'_template -all'));
            num_sess = num_sess + 1;
        end
    end
%     for jj = 1:num_sess
%         long_cmds{jj} = strcat('recon-all -long', strcat([' ',ID],'_',num2str(jj)),...
%             strcat([' ',ID],'_template -all'));
%     end
    long_cmds = long_cmds(1:(num_sess-1));
end