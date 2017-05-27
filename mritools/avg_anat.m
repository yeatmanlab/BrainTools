%%
sub = {'NLR_205_AC'};

workingDir = '/home/sjjoo/git/BrainTools/mritools';
% Change filelist per subject
% ATTN: If acpc aligned anatomy has been created, we should set the first
% filelist as the acpc aligned one in the 'anatomy' folder. And set the
% 'alignLandmarks' as false so the function below tries to align the
% subsequent images to the first one in the list assuming that is acpc
% aligned (which is true).
%
% Otherwise, just list mprage files in the MRI folder and set the
% 'alignLandmarks' as [] so the function below bring up the gui to acpc
% align and then average.

%%
for ss = 1:length(sub)
    cd(workingDir)
    outpath = sprintf('/home/sjjoo/analysis/anatomy/%s',sub{ss});
    if isempty(dir(outpath))
        mkdir(outpath);
    end
    
    sessions = getsessions(strcat('/mnt/diskArray/projects/MRI/',sub{ss}));
    
    % find raw par/rec files, make a compressed nifti and store in the
    % subject's 'raw' folder
    % start with the second session, since the first is already
    % acpc-aligned and stored in the anatomy folder
    filelist = {}; namelist = {};
    for ii = 2:numel(sessions)
        mkdir(sprintf('%s/%s', outpath, sessions{ii}));
        cd(fullfile('/mnt/diskArray/projects/MRI/',sub{ss},sessions{ii},'raw'))
        temp = dir(fullfile(pwd, '*VBM*.PAR'));
        if ~isempty(temp)
            parlist = fullfile('/mnt/diskArray/projects/MRI/',sub{ss},sessions{ii},'raw',temp(numel(temp)).name);
            system(sprintf('parrec2nii -c --overwrite -o %s %s',sprintf('%s/%s',outpath,sessions{ii}),parlist));
            [PATHSTR,NAME,EXT] = fileparts(parlist);
    %         filelist{ii} = fullfile(PATHSTR, strcat(NAME, '.nii.gz'));
            filelist{ii} = sprintf('%s/%s', sprintf('%s/%s', outpath, sessions{ii}), strcat(NAME, '.nii.gz'));
            namelist{ii} = NAME;
    %         temp = readFileNifti(filelist{ii});
    %         temp = niftiCheckQto(temp);
    %         delete(filelist{ii})
    %         niftiWrite(temp, filelist{ii});--out_orientation
            system(sprintf('mri_convert --out_orientation RAS --out_type nii %s %s', filelist{ii}, filelist{ii}));
        end
    end
    
    filelist{1} = strcat(outpath,'/t1_acpc.nii.gz');
    namelist{1} = 't1_acpc.nii.gz';
    
    % Let's check if there is an empty cell
    idx = [];
    for i = 1: length(filelist)
        if isempty(filelist{i})
            idx = [idx i];
        end
    end
    filelist(idx) = [];
    
    voxres = [0.8, 0.8, 0.8];
    
    keepfile = [];
    for i = 1:length(filelist)
        temp = readFileNifti(filelist{i});
        tempmontage = makeMontage(temp.data,100:220);
        figure
        imagesc(tempmontage), colormap('gray')
        drawnow;
        % type 1 to include the file, or click on the figure to exclude
        keepfile(i) = waitforbuttonpress
    end
    
    %% align and average files
    gind = find(keepfile); % just include the sessions coded with a 1
    
    filelist = filelist(gind);
%     filelist = horzcat(strcat(outpath,'/t1_acpc.nii.gz'), filelist)    
    % outImg = mrAnatAverageAcpcNifti(fileNameList, outFileName, [alignLandmarks=[]], [newMmPerVox=[1 1 1]], [weights=ones(size(fileNameList))], [bb=[-90,90; -126,90; -72,108]'], [showFigs=true], [clipVals])
    outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), false, voxres(1:3));
    
    close all
end

