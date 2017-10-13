
% function bde_preprocess_anatomy_CT(subject, session, clobber, makesurface, maindir, anatdir)
% bde_preprocess_anatomy(subject, session, clobber, makesurface, maindir, anatdir)
%
% Checks whether anatomical pre-processing has been done for
% a given subject and performs (1) ACPC alignment and (2) cortical surface
% reconstruction as needed.
%
% Inputs:
%   subject should be NLR_<subject#>_<initials>
%   session is visit #, 1 through 4 for most LMB subjects (to do - update for
% multiple inputs to average over sessions)
%   If clobber = 1, overwrite existing anatomical images and rerun alignment
% and segmentation/reconstruction in Freesurfer.

% Assumes you have: vistasoft repo, Freesurfer installed in /usr/local
%% setup directory info
subID = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH',...
    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_127_AM', ...
    'NLR_105_BB', 'NLR_132_WP', 'NLR_187_NB', 'RI_124_AT', 'RI_143_CH', ...
    'RI_138_LA', 'RI_141_GC', 'RI_144_OL','NLR_199_AM', 'NLR_130_RW', ...
    'NLR_133_ML', 'NLR_146_TF', 'NLR_195_AW', 'NLR_191_DF', 'NLR_197_BK', ...
    'NLR_201_GS', 'NLR_202_DD', 'NLR_203_AM', 'NLR_101_LG', 'NLR_103_AC'};
clobber = 1;
makesurface = 0;
maindir = '/mnt/scratch/MRI/';
anatdir = '/mnt/scratch/anatomy/';
addpath(genpath('/mnt/scratch'))

%%
for ss = 26%:numel(subID)
    subject = subID{ss};
    % Find the folders labeled by date and pick desired session:
    allsessions = dir(fullfile(maindir, subject));
    %   (session folders have form yyyy/mm/dd, so length is 8 chars)
    allsessions = allsessions(cellfun(@length, {allsessions.name})==8);

    % Skip sessions which do not have reference (avg) file
    %if ~exist(fullfile(anatdir,subject,'t1_acpc_avg.nii.gz'), 'file')
    %    continue
    %end

    for session = 4%1:numel(allsessions)
        % Breaks out for session missing anatomy data
        if (ss == 16 && session == 3)
            break
        end
        sessiondir = allsessions(session).name;

        rawdir = fullfile(maindir, subject, sessiondir, 'raw');
        cd(rawdir)

        T1path = dir(fullfile(rawdir, '*VBM*.nii.gz'));

        % Convert PAR/REC files to make nifti if it doesn't exist yet
        if isempty(T1path)
            cmd = sprintf('parrec2nii -b -c --scaling=%s --store-header --output-dir=%s --overwrite %s', ...
                'dv', rawdir, '*VBM*.PAR');
            system(cmd) % convert_parrec(cellstr(parfiles), rawdir);
            T1path = dir(fullfile(rawdir, '*VBM*.nii.gz'));
        end

        % Continue if the session contains VBM files
        % if (~isempty(T1path)) ... end

        T1path = fullfile(rawdir,T1path(1).name);
        T1path = mri_rms(T1path);

        % Read in nifti file and update header as needed
        im = niftiRead(T1path);
        %
        im = niftiCheckQto(im);
        niftiWrite(im, T1path);

        im = niftiRead(T1path);

        % Get the voxel resolution of the image (mm)
        voxres = [.8,.8,.8]; % voxres = diag(im.qto_xyz)';

        % ACPC align and resample
        if ~exist(fullfile(anatdir, subject), 'dir')
            mkdir(fullfile(anatdir, subject))
        end

        % make sure that raw images have RAS orientation, since that's the
        % orientation that our template image has; otherwise, the output image
        % may be flipped left-right
        system(sprintf('mri_convert --out_orientation RAS --out_type nii %s %s', T1path, T1path));
        % load in the average template file as the first input image, then
        % the raw T1 will be aligned properly (sending in the template as a
        % separate argument doesn't properly align the two)
        mrAnatAverageAcpcNifti({fullfile(anatdir,subject,'t1_acpc_avg.nii.gz'), T1path}, ...
            fullfile(anatdir, subject, strcat('t1_acpc_',num2str(session),'.nii.gz')), ...
            0, voxres(1:3),[],[],0);
        close all

    end
end

%%
% mrAnatAverageAcpcNifti({T1path}, {'/home/mpenrod/t1_acpc_avg.nii.gz'}, voxres(1:3));

%
% % Make Freesurfer surfaces
%if (~exist(fullfile('/usr/local/freesurfer/subjects', subject), 'dir') || clobber==1) && makesurface == 1
% Remove Freesurfer directory if clobbering
%if exist(fullfile('/usr/local/freesurfer/subjects', subject), 'dir')
%   rmdir (fullfile('/usr/local/freesurfer/subjects', subject), 'dir')
%end

% Iterates through subjects and processed session images and recons them
% for ss = 1%:numel(subID)
%     subject = subID{ss};
%     for processed = 2:5%1:5
%         file = fullfile(anatdir, subject, (strcat('t1_acpc_', num2str(processed), '.nii.gz')));
%         if exist(file,'file')
%             cmd = sprintf('recon-all -i %s -subjid %s -all', file, strcat(subject, '_', num2str(processed)));
%             system(cmd);
%             % reformat freesurfer segmentation for mrVista ITKgray
%             fs_ribbon2itk(strcat(subject,'_',num2str(processed)), fullfile(anatdir,subject,...
%                 (strcat('fs_seg_', num2str(processed),'.nii.gz'))),[], file)
%         else
%             break
%         end
%     end
% end

% break
%
% fname = fullfile(anatdir,subject,'fs_seg.nii.gz');
% msh = meshBuildFromClass(fname, [], 'right');
% msh = meshSmooth(msh);
% msh = meshColor(msh);
%
% % smooth it
% msh = meshSet(msh, 'smooth_iterations', 200);
% msh = meshSet(msh, 'smooth_relaxation', 0.5);
% msh = meshSet(msh, 'smooth_sinc_method', 1);
%
% % load up visualization
% meshVisualize(msh);
%
% save(fullfile(anatdir,subject,'left200smooth'), 'msh');
% cd(fullfile(anatdir,subject))
% mrmInflate('left200smooth', 200);
%
%
% meshVisualize(msh);





