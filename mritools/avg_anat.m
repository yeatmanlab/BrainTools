
sub = {'NLR_110_HH', 'NLR_145_AC', 'NLR_150_MG', 'NLR_151_RD', 'NLR_152_TC', ...
    'NLR_160_EK', 'NLR_161_AK', 'NLR_162_EF',  'NLR_163_LF','NLR_164_SF', ...
    'NLR_170_GM','NLR_172_TH','RI_124_AT', 'RI_143_CH', ...
    'RI_138_LA', 'RI_141_GC', 'RI_144_OL','NLR_199_AM', 'NLR_130_RW', ...
    'NLR_133_ML', 'NLR_146_TF', 'NLR_195_AW'};


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

for ss = 1:length(sub)
    
    outpath = sprintf('/home/ehuber/analysis/anatomy/%s',sub{ss});
    if isempty(dir(outpath))
        mkdir(outpath);
    end
    
    sessions = getsessions(strcat('/home/ehuber/analysis/MRI/',sub{ss}));
    
    % find raw par/rec files, make a compressed nifti and store in the
    % subject's 'raw' folder
    % start with the second session, since the first is already
    % acpc-aligned and stored in the anatomy folder
    filelist = {};
    for ii = 2:numel(sessions)
        cd(fullfile('/home/ehuber/analysis/MRI/',sub{ss},sessions{ii},'raw'))
        temp = dir(fullfile(pwd, '*VBM*.PAR'));
        parlist = fullfile('/home/ehuber/analysis/MRI/',sub{ss},sessions{ii},'raw',temp.name);
        system(sprintf('parrec2nii -c -o %s %s',pwd,parlist));
        [PATHSTR,NAME,EXT] = fileparts(parlist);
        filelist{ii} = fullfile(PATHSTR, strcat(NAME, '.nii.gz'));
        namelist{ii} = NAME;
        temp = readFileNifti(filelist{ii});
        temp = niftiCheckQto(temp);
        delete(filelist{ii})
        niftiWrite(temp, filelist{ii});
    end
    
    filelist{1} = strcat(outpath,'/t1_acpc.nii.gz');
    namelist{1} = 't1_acpc.nii.gz';
    
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
    
    %%
    % UPDATE this so Only include good session numbers here!!!
    gind = find(keepfile); % just include the sessions coded with a 1
    
    filelist = filelist(gind);
%     filelist = horzcat(strcat(outpath,'/t1_acpc.nii.gz'), filelist)    
    % outImg = mrAnatAverageAcpcNifti(fileNameList, outFileName, [alignLandmarks=[]], [newMmPerVox=[1 1 1]], [weights=ones(size(fileNameList))], [bb=[-90,90; -126,90; -72,108]'], [showFigs=true], [clipVals])
    outImg = mrAnatAverageAcpcNifti(filelist, sprintf('%s/t1_acpc_avg.nii.gz',outpath), false, voxres(1:3));
    
end

