function BIDS_copyraw(basedir, dirstring, include, exclude, outbasedir)
%
%
% example:
%
% basedir = '/mnt/diskArray56/projects/MRI'
% dirstring = 'NLR_'
% exclude = {'NLR_003' 'NLR_006' 'NLR04' 'NLR_TestScans'}
% include = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH', ...
%    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
%    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
%    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
%    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_127_AM', ...
%    'NLR_105_BB', 'NLR_132_WP'};
% outbasedir = '/mnt/scratch/BIDS_DATA'
% BIDS_copyraw(basedir, dirstring, include, exclude, outbasedir)

%% Find data

% List directory
d = dir(fullfile(basedir,[dirstring '*']));

% Only include designated subjects
keep = zeros(length(d),1);
for ii = 1:length(d)
    if any(strcmp(d(ii).name,include))
        keep(ii) = 1;
    end
end
d = d(logical(keep));

% Remove subjects designated in exclude
keep = ones(length(d),1);
for ii = 1:length(d)
    if any(strcmp(d(ii).name,exclude))
        keep(ii) = 0;
    end
end
d = d(logical(keep));

ss = 0;
% Record all subject session directories
for ii = 1:length(d)
    subdirs{ii} = d(ii).name;
    dss = dir(fullfile(basedir, subdirs{ii}));
    
    for jj  = 3:length(dss)
        % Check if there is a raw data directory
        if exist(fullfile(basedir, subdirs{ii}, dss(jj).name, 'raw'), 'dir')
            ss = ss+1;
            sessdirs_subject{ss} = subdirs{ii};
            sessdirs_session{ss} = dss(jj).name;
        end
    end
end

% Table with all the subject-session directory combinations
sessdirs = table(sessdirs_subject(:),sessdirs_session(:), 'variableNames',{'subject', 'session'})

%% Copy data and save to BIDS
dmrifilelist = table; t1filelist = table;
for ii = 1:size(sessdirs,1)
    subdir = fullfile(outbasedir, sessdirs.subject{ii});
    sessdir = fullfile(subdir,sessdirs.session{ii});
    dwidir = fullfile(sessdir, 'raw');
    anatdir = fullfile(sessdir, 'anat');
    rawdir = fullfile(basedir,sessdirs.subject{ii},sessdirs.session{ii},'raw')
    
    % Raw diffusion files - .PAR and .REC
    diffusionpath = dir(fullfile(rawdir, '*DWI*'));
    dmrifiles = {diffusionpath(:).name};
    if ~isempty(dmrifiles),mkdir(dwidir);end
    for jj = 1:length(dmrifiles)
        copyfile(fullfile(rawdir,dmrifiles{jj}),fullfile(dwidir,dmrifiles{jj}));
        dmrifilelist = [dmrifilelist; table(sessdirs.subject(ii), sessdirs.session(ii), {dwidir},dmrifiles(jj),...
            'variableNames',{'subject', 'session', 'dwidir', 'filename'})];
    end
    
    % Raw T1 anatomy files - .PAR and .REC
    t1path = dir(fullfile(rawdir, '*MEMP_VBM_SENSE*'));
    t1files = {t1path(:).name};
    if ~isempty(t1path),mkdir(anatdir);end
    for jj = 1:length(t1files)
        copyfile(fullfile(rawdir,t1files{jj}),fullfile(anatdir,t1files{jj}));
        t1filelist = [t1filelist; table(sessdirs.subject(ii), sessdirs.session(ii), {anatdir},t1files(jj),...
            'variableNames',{'subject', 'session', 'anatdir', 'filename'})];
    end
    
    
end

save BIDS_copyraw_log dmrifilelist t1filelist sessdirs

return

%%

if isempty(diffusionpath)
    cmd = sprintf('parrec2nii --bvs -c --scaling=%s --store-header --output-dir=%s --overwrite %s', ...
        'dv', rawdir, '*DWI*.PAR');
    system(cmd) % convert_parrec(cellstr(parfiles), rawdir);
    diffusionpath = dir(fullfile(rawdir, '*DWI*.nii.gz'));
end
