function BIDS_copyraw(basedir, dirstring, include, exclude, outbasedir, ...
    dwidirname, rawdirname, dmristring, t1string, bvalall, dcm2nii, fsdir, fssubprefix, t1basedir, parreconly)
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

if ~exist('dwidirname','var') || isempty(dwidirname)
    dwidirname = 'dwi';
end
if ~exist('rawdirname','var') || isempty(rawdirname)
    rawdirname = 'raw';
end
if ~exist('dmristring','var') || isempty(dmristring)
    dmristring = '*DWI*';
end
if ~exist('dcm2nii','var') || isempty(dcm2nii)
    dcm2nii = 0;
end
if ~exist('parreconly','var') || isempty(parreconly)
    parreconly=0;
end
% List directory
d = dir(fullfile(basedir,[dirstring '*']));

% Only include designated subjects
if exist(include,'var') && ~isempty(include)
    keep = zeros(length(d),1);
    for ii = 1:length(d)
        if any(strcmp(d(ii).name,include))
            keep(ii) = 1;
        end
    end
    d = d(logical(keep));
end

% Remove subjects designated in exclude
if exist('exclude','var') && ~isempty(exclude)
    keep = ones(length(d),1);
    for ii = 1:length(d)
        if any(strcmp(d(ii).name,exclude))
            keep(ii) = 0;
        end
    end
    d = d(logical(keep));
end

ss = 0;
% Record all subject session directories
for ii = 1:length(d)
    subdirs{ii} = d(ii).name; % Subject directory
    dss = dir(fullfile(basedir, subdirs{ii})); % Session directories
    
    for jj  = 3:length(dss)
        % Check if there is a raw data directory
        if exist(fullfile(basedir, subdirs{ii}, dss(jj).name, rawdirname), 'dir')
            ss = ss+1;
            sessdirs_subject{ss} = subdirs{ii};
            sessdirs_session{ss} = dss(jj).name;
        end
    end
end

% Table with all the subject-session directory combinations
sessdirs = table(sessdirs_subject(:),sessdirs_session(:), 'variableNames',{'subject', 'session'});

%% Copy data and save to BIDS
dmrifilelist = table; t1filelist = table;
for ii = 1:size(sessdirs,1)
    % Check if the subdir already contains 'sub-'
    if strfind(sessdirs.subject{ii},'sub-') == 1
        subdir = fullfile(outbasedir, sessdirs.subject{ii});
    else
        subdir = fullfile(outbasedir, ['sub-' sessdirs.subject{ii}]);
    end
    sessdir = fullfile(subdir,sessdirs.session{ii});
    dwidir = fullfile(sessdir, dwidirname);
    anatdir = fullfile(sessdir, 'anat');
    rawdir = fullfile(basedir,sessdirs.subject{ii},sessdirs.session{ii},rawdirname);
    
    % Raw diffusion files - .PAR and .REC
    diffusionpath = dir(fullfile(rawdir, dmristring));
    dmrifiles = {diffusionpath(:).name};
    % Only continue if this subject has dwi data
    if ~isempty(dmrifiles) && ~exist(dwidir, 'dir')
        mkdir(dwidir);
    else
        continue; % Skip sub if no dwi
    end
    for jj = 1:length(dmrifiles)
        % Try to parse bvalue if it wasn't supplied
        if ~exist('bvalall','var') || isempty(bvalall)
            b(1) = strfind(dmrifiles{jj},'_b'); b(2) = strfind(dmrifiles{jj},'_SSGR');
            bval = dmrifiles{jj}(b(1)+2 : b(2)-1);
        elseif isnumeric(bvalall)
            bval = num2str(bvalall);
        else
            bval = bvalall;
        end
        % Check if the session name already contains 'ses'
        if strfind(sessdirs.session{ii},'ses') == 1
            sesname = sessdirs.session{ii};
        else
            sesname = sprintf('ses-%s',sessdirs.session{ii});
        end
        [~,~,ext] = fileparts(dmrifiles{jj});
        if parreconly ==1 && ~strcmp(ext,'.REC') && ~strcmp(ext,'.PAR')
            continue
        end
        dmrioutfile = fullfile(dwidir,sprintf('sub-%s_%s_acq-b%s_dwi%s',sessdirs.subject{ii}, sesname, bval, ext));
        copyfile(fullfile(rawdir,dmrifiles{jj}),dmrioutfile);
        if dcm2nii == 1 && strcmp(ext,'.REC')
            cmd = sprintf('dcm2nii -o %s -f Y -d N -e N -p N %s',fileparts(dmrioutfile),dmrioutfile);
            system(cmd);
        end
        dmrifilelist = [dmrifilelist; table(sessdirs.subject(ii), sessdirs.session(ii), {dwidir},dmrifiles(jj),{dmrioutfile},...
            'variableNames',{'subject', 'session', 'dwidir', 'filein', 'fileout'})];
    end
    
    % Raw T1 anatomy files - .PAR and .REC
    % Only if defined in t1string
    if exist('t1string','var') && ~isempty(t1string)
        t1path = dir(fullfile(rawdir, t1string));
        t1files = {t1path(:).name};
        if ~isempty(t1path),mkdir(anatdir);end
        for jj = 1:length(t1files)
            copyfile(fullfile(rawdir,t1files{jj}),fullfile(anatdir,t1files{jj}));
            [~,~,ext] = fileparts(t1files{jj});
            if parreconly ==1 && ~strcmp(ext,'.REC') && ~strcmp(ext,'.PAR')
                continue
            end
            if dcm2nii == 1 && strcmp(ext,'.REC')
                cmd = sprintf('dcm2nii -o %s -f Y -d N -e N -p N %s',anatdir,fullfile(anatdir,t1files{jj}));
                system(cmd);
            end
            t1filelist = [t1filelist; table(sessdirs.subject(ii), sessdirs.session(ii), {anatdir},t1files(jj),...
                'variableNames',{'subject', 'session', 'anatdir', 'filename'})];
        end
    else
        t1filelist = [];
    end
   
end

save(fullfile(outbasedir,'BIDS_copyraw_log'), 'dmrifilelist', 't1filelist', 'sessdirs')

% Freesurfer
if exist('fsdir','var') && ~isempty(fsdir)
    if ~exist('fssubprefix', 'var')
        fssubprefix=[];
    end
    subdirs = unique(dmrifilelist.subject);
    for ii = 1:length(subdirs)
        if strfind(subdirs{ii},'sub-') == 1
            subdir = subdirs{ii};
        else
            subdir = ['sub-' subdirs{ii}];
        end
        fsoutdir = fullfile(outbasedir,'derivatives',subdir,'freesurfer');
        if ~exist(fileparts(fileparts(fsoutdir)),'dir'), mkdir(fileparts(fileparts(fsoutdir))); end
        if ~exist(fileparts(fsoutdir),'dir'), mkdir(fileparts(fsoutdir)); end
        if ~exist(fsoutdir,'dir'), mkdir(fsoutdir); end  

        fsindir = fullfile(fsdir,[fssubprefix subdirs{ii}]);
        if exist(fsindir,'dir')
            fprintf('\n%s Copying FREESURFER\n',fsindir)
            copyfile(fsindir,fsoutdir);
        else
            fprintf('\n%s DOES NOT EXIST\n',fsindir)
        end
        % Copy t1 associated with fs run
        if exist('t1basedir','var') && ~isempty(t1basedir)
            t1filepath = fullfile(t1basedir,[fssubprefix subdirs{ii}],t1string);
            copyfile(t1filepath,fullfile(fsoutdir,t1string));
            
        end
    end
end

return

%%

if isempty(diffusionpath)
    cmd = sprintf('parrec2nii --bvs -c --scaling=%s --store-header --output-dir=%s --overwrite %s', ...
        'dv', rawdir, '*DWI*.PAR');
    system(cmd) % convert_parrec(cellstr(parfiles), rawdir);
    diffusionpath = dir(fullfile(rawdir, '*DWI*.nii.gz'));
end
