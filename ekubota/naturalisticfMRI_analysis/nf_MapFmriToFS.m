%% Set up subject list and directory structure
addpath('/usr/local/freesurfer/matlab/')
fsdir = '/mnt/scratch/projects/freesurfer/';
prekdir = '/mnt/scratch/PREK_Analysis/';
fsavgout = '/mnt/scratch/projects/freesurfer/fsaverage_maps/nf';
subs = {'PREK_1112','PREK_1676','PREK_1691','PREK_1715','PREK_1762',...
    'PREK_1901','PREK_1916','PREK_1951','PREK_1964'};
for ii = 1:length(subs)
    mapdirs{ii} = fullfile(fsdir,subs{ii},'maps');
    if ~exist(mapdirs{ii},'dir'), mkdir(mapdirs{ii});end
end

%% Map data to freesurfer
cons = {'reliabilityCorrelation.nii.gz', 'AllvBaseline.nii.gz', 'TextvNontext.nii.gz'};
consout = {'reliabilityCorrelation_s3.nii.gz', 'AllvBaseline_s3.nii.gz', 'TextvNontext_s3.nii.gz'};

for ii = 1:length(subs)
    for cc = 1:length(cons)
        im = niftiRead(fullfile(prekdir, subs{ii}, 'nf_anatomy',cons{cc}));
        im.data = smooth3(im.data,'gaussian',[3 3 3]);
        im.fname = fullfile(prekdir, subs{ii}, 'nf_anatomy',consout{cc});
        niftiWrite(im);
    end
end
for ii = 1:length(subs)
    
    % Reliability (lh and rh)
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'reliabilityCorrelation_s3.nii.gz'),...
        subs{ii},'lh',fullfile(mapdirs{ii}, 'lh_reliabilityCorrelation.mgz'));
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'reliabilityCorrelation_s3.nii.gz'),...
        subs{ii},'rh',fullfile(mapdirs{ii}, 'rh_reliabilityCorrelation.mgz'));
    
    % All v Baseline (lh and rh)
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'AllvBaseline_s3.nii.gz'),...
        subs{ii},'lh',fullfile(mapdirs{ii}, 'lh_AllvBaseline.mgz'));
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'AllvBaseline_s3.nii.gz'),...
        subs{ii},'rh',fullfile(mapdirs{ii}, 'rh_AllvBaseline.mgz'));
    
    % Text V No Text (lh and rh)
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'TextvNontext_s3.nii.gz'),...
        subs{ii},'lh',fullfile(mapdirs{ii}, 'lh_TextvNontext.mgz'));
    fs_vol2surf(fullfile(prekdir, subs{ii}, 'nf_anatomy', 'TextvNontext_s3.nii.gz'),...
        subs{ii},'rh',fullfile(mapdirs{ii}, 'rh_TextvNontext.mgz'));
    
end

%% Transform individual maps to fsaverage

for ii = 1:length(subs)
    
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'lh_reliabilityCorrelation.mgz'),...
        fullfile(fsavgout, sprintf('lh_%s_reliabilityCorrelation.mgz',subs{ii})), 'lh');
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'rh_reliabilityCorrelation.mgz'),...
        fullfile(fsavgout, sprintf('rh_%s_reliabilityCorrelation.mgz',subs{ii})), 'rh');
    
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'lh_AllvBaseline.mgz'),...
        fullfile(fsavgout, sprintf('lh_%s_AllvBaseline.mgz',subs{ii})), 'lh');
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'rh_AllvBaseline.mgz'),...
        fullfile(fsavgout, sprintf('rh_%s_AllvBaseline.mgz',subs{ii})), 'rh');
    
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'lh_TextvNontext.mgz'),...
        fullfile(fsavgout, sprintf('lh_%s_TextvNontext.mgz',subs{ii})), 'lh');
    fs_surf2surf(subs{ii}, 'fsaverage', fullfile(mapdirs{ii}, 'rh_TextvNontext.mgz'),...
        fullfile(fsavgout, sprintf('rh_%s_TextvNontext.mgz',subs{ii})), 'rh');
end

%% Group level statistics
mapnames= {'lh_PREK_*_AllvBaseline.mgz','rh_PREK_*_AllvBaseline.mgz',...
    'lh_PREK_*_reliabilityCorrelation.mgz', 'rh_PREK_*_reliabilityCorrelation.mgz',...
    'lh_PREK_*_TextvNontext.mgz','rh_PREK_*_TextvNontext.mgz'};
outnames= {'lh_PREK_AllvBaseline.mgz','rh_PREK_AllvBaseline.mgz',...
    'lh_PREK_reliabilityCorrelation.mgz', 'rh_PREK_reliabilityCorrelation.mgz',...
    'lh_PREK_TextvNontext.mgz','rh_PREK_TextvNontext.mgz'};

for m = 1:length(mapnames)
    mgz = dir(fullfile(fsavgout,mapnames{m}));
    
    for ii = 1:length(allmgz)
        [con(:,ii) M{ii}] = load_mgh(fullfile(fsavgout,mgz(ii).name));
    end
    [~,~,~,stats] = ttest(con');
    T = stats.tstat';
    save_mgh(T,fullfile(fsavgout,sprintf('Tstat_%s',outnames{m})),M{1});
end