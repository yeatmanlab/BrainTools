fsldirs = {'/mnt/diskArray/projects/KNK/data/20140814S015/fsl_84dir'...
    '/mnt/diskArray/projects/KNK/data/20140810S018/fsl_84dir'...
    '/mnt/diskArray/projects/KNK/data/20140721S017/fsl_84dir'...
    '/mnt/diskArray/projects/KNK/data/20140707S019/fsl_84dir'}
fsdirs  = {'/mnt/diskArray/projects/KNK/data/S015'...
    '/mnt/diskArray/projects/KNK/data/S018'...
    '/mnt/diskArray/projects/KNK/data/S017'...
    '/mnt/diskArray/projects/KNK/data/S019'}

for ii = 1:length(fsldirs)
    % Compute registration between fsl and freesurfer
    fsl_register2freesurfer(fsldirs{ii},fsdirs{ii});
    % Paths to vwfa and ips rois
    roi1_fs = fullfile(fsdirs{ii},'mri','rois','LH_VWFA1.nii.gz');
    roi2_fs = fullfile(fsdirs{ii},'mri','rois','LH_IPS.nii.gz');
    % register roi 2 fsl data
    famap = fullfile(fsldirs{ii},'dtifit','dti_FA.nii.gz');
    xform_fs2fa = fullfile(fsldirs{ii},'dtifit','struct2fa.mat');
    xform_fa2fs = fullfile(fsldirs{ii},'dtifit','fa2struct.mat');
    if ~exist(fullfile(fsldirs{ii},'dtifit','rois'),'dir')
        mkdir(fullfile(fsldirs{ii},'dtifit','rois'));
    end
    roi1_fsl = fullfile(fsldirs{ii},'dtifit','rois','LH_VWFA1.nii.gz');
    roi2_fsl = fullfile(fsldirs{ii},'dtifit','rois','LH_IPS.nii.gz');
    cmd = sprintf('flirt -v -in %s -ref %s -applyxfm -init %s -out %s',roi1_fs, famap, xform_fs2fa, roi1_fsl);
    system(cmd);
    cmd = sprintf('flirt -v -in %s -ref %s -applyxfm -init %s -out %s', roi2_fs, famap, xform_fs2fa, roi2_fsl);
    system(cmd);
    % run probtrack
    bedpostx = fullfile(fsldirs{ii},'eddy.bedpostX','merged');
    bmask = fullfile(fsldirs{ii},'eddy.bedpostX','nodif_brain_mask');
    ptrackout = fullfile(fsldirs{ii},'eddy.bedpostX','probtrack','LH_VWFAtoIPS');
    cmd = sprintf('probtrackx2  -x %s  -l --onewaycondition -c 0.2 --steplength=0.5 --nsamples=5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --forcedir --opd -s %s -m %s  --dir=%s --waypoints=%s  --waycond=AND',...
        roi1_fsl,bedpostx,bmask,ptrackout,roi2_fsl);
%     cmd = sprintf('probtrackx2  -x %s  -l --onewaycondition -c 0.2 --steplength=0.5 --nsamples=5000 --fibthresh=0.01 --distthresh=0.0 --sampvox=0.0 --forcedir --opd -s %s -m %s  --dir=%s',...
%         roi1_fsl,bedpostx,bmask,ptrackout);
    system(cmd)
    % Transform probtrack to freesurfer
    ptrackoutnii = fullfile(ptrackout,'fdt_paths.nii.gz');
    [~,fs_sub] = fileparts(fsdirs{ii});
    ptrack_fs    = fullfile(fsdirs{ii},'mri','rois','LH_VWFAtoIPS.nii.gz');
    fs_anat      = fullfile(fsdirs{ii},'mri','nifti','brain.nii.gz')
    cmd = sprintf('flirt -in %s -ref %s -applyxfm -init %s -out %s',ptrackoutnii, fs_anat, xform_fa2fs, ptrack_fs);
    system(cmd)
    % map to surface
    ptrack_surf = [ptrack_fs(1:end-6) 'mgh'];
    cmd = sprintf('mri_vol2surf --mov %s --regheader %s --o %s --hemi lh --surf white --projfrac 0',ptrack_fs,fs_sub,ptrack_surf);
    system(cmd);
    % map to average surface
    surf_aligned{ii} = [ptrack_surf(1:end-4) '_fsavg.mgh'];
    cmd = sprintf('mri_surf2surf --srcsubject %s --srcsurfval %s --trgsubject fsaverage --hemi lh --trgsurfval %s',...
        fs_sub, ptrack_surf, surf_aligned{ii});
    system(cmd);
end

% Average fiber density maps
cmd = sprintf('mri_concat --i %s %s %s %s --o %s --mean',surf_aligned{:},'/mnt/diskArray/projects/KNK/data/fsaverage/maps/LH_VWFAtoIPS_avg.mgh');
system(cmd);

