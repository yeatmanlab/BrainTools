function LB_antsAnalysis

load /biac4/wandell/data/Lindamood_Bell/MRI/analysis/AFQ_sge_16-Jan-2014.mat;
outdir = '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/maps';
fid = fopen('/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/maps/antsScript.txt','w');
fprintf(fid,'buildtemplateparallel.sh -d 3 -o FATEMP -n 0');
for ii = 1:length(afq.sub_dirs)
    dt = dtiLoadDt6(fullfile(afq.sub_dirs{ii},'dt6.mat'));
    b0 = readFileNifti(dt.files.b0);
    b0.fname = fullfile(outdir,sprintf('%d_b0.nii.gz',ii));
    writeFileNifti(b0);
    [fa,md,rd,ad] = dtiComputeFA(dt.dt6);
    % write out md map
    b0.data = md;
    b0.cal_max = 4;
    b0.fname = fullfile(outdir,sprintf('%d_md.nii.gz',ii));
    writeFileNifti(b0);
    % fa
    b0.data = fa;
    b0.cal_max=1;
    b0.fname = fullfile(outdir,sprintf('%d_fa.nii.gz',ii));
    writeFileNifti(b0);
    fprintf(fid,sprintf(' %d_fa.nii.gz',ii));
end
fclose(fid)

return
load /biac4/wandell/data/Lindamood_Bell/MRI/analysis/AFQ_sge_16-Jan-2014.mat;
outdir = '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/maps';
cd(outdir)
fname = '/biac4/wandell/data/Lindamood_Bell/MRI/child/intervention/maps/antsScript_register.txt';
fid = fopen(fname,'w');
for ii = 1:length(afq.sub_dirs)
    fprintf(fid,sprintf('\nantsIntroduction.sh -d 3 -r GR_iteration_0/FATEMPtemplate.nii.gz -i %d_fa.nii.gz -n 0 -m 60x180x50',ii));
    fprintf(fid,'\nWarpImageMultiTransform 3 %d_fa.nii.gz %d_fadeformed.nii.gz -R GR_iteration_0/FATEMPtemplate.nii.gz %d_faWarp.nii.gz %d_faAffine.txt',ii,ii,ii,ii);
    fprintf(fid,'\nWarpImageMultiTransform 3 %d_md.nii.gz %d_mddeformed.nii.gz -R GR_iteration_0/FATEMPtemplate.nii.gz %d_faWarp.nii.gz %d_faAffine.txt',ii,ii,ii,ii);
    
end
fclose(fid)
system(sprintf('chmod 777 %s',fname))
system(sprintf('%s',fname));

%% Analyze voxelwise
cd(outdir)
X = [-3,1; -1, 1; 1 ,1;3,1];
x = [-3 -1 1 3];
slopeIm = [];
property = 'md'
for ii = 1:4:24
    % image numbers
    n = ii:ii+3;
    % load images into a 4d array
    c = 0;
    for jj = n
        c = c+1;
       im(c) = readFileNifti(sprintf('%d_%sdeformed.nii.gz',jj,property));
       im(c).data = smooth3(im(c).data,'gaussian',[3 3 3]);
    end
    % Fit a line to every voxel
    cim = zeros(size(im(1).data));
    % mask
    mask = find(im(1).data~=0 | im(2).data~=0 | im(3).data~=0 | im(4).data~=0)';
    tic
    for k = mask
        %p = polyfit(x,[im(1).data(k) im(2).data(k) im(3).data(k) im(4).data(k)],1);
        p = X\[im(1).data(k) im(2).data(k) im(3).data(k) im(4).data(k)]';
        cim(k) = p(1);
    end
    toc
    slopeIm = cat(4,slopeIm,cim);
end

% compute t stat
m = mean(slopeIm,4);
se = std(slopeIm,[],4)./sqrt(size(slopeIm,4));
Timg = m./se;
% write tmap
tmap = im(1);
tmap.fname = sprintf('Tmap_%s_pos.nii.gz',property);
tmap.data=Timg;
writeFileNifti(tmap);
tmap.data=tmap.data.*-1;
tmap.fname = sprintf('Tmap_%s_neg.nii.gz',property);
writeFileNifti(tmap);

