%% setup directory info
subID = {'NLR_101_LG', 'NLR_103_AC'};
clobber = 1;
makesurface = 0;
maindir = '/mnt/scratch/MRI/';
anatdir = '/mnt/scratch/anatomy/';
addpath(genpath('/mnt/scratch'))

for ss = 1:numel(subID)
    subject = subID{ss};
    % Find the folders labeled by date and pick desired session:
    allsessions = dir(fullfile(maindir, subject));
    %   (session folders have form yyyy/mm/dd, so length is 8 chars)
    allsessions = allsessions(cellfun(@length, {allsessions.name})==8);
    
    % Skip sessions which do not have reference (avg) file
    %if ~exist(fullfile(anatdir,subject,'t1_acpc_avg.nii.gz'), 'file')
    %    continue
    %end
    
    for session = 1:numel(allsessions)
        
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
       
        T1path = fullfile(rawdir,T1path(1).name);
        % T1path = mri_rms(T1path);
        
        im = niftiRead(T1path); % Read root mean squared image
        voxres = [.8,.8,.8]; % voxres = diag(im.qto_xyz)';
        mrAnatAverageAcpcNifti({T1path}, fullfile(anatdir, subject, strcat('t1_acpc_',num2str(session),'.nii.gz')), [], voxres(1:3))
    end
end