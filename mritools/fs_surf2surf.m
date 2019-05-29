function cmd = fs_surf2surf(srcsubj, trgsubj, surf, outfile, hemi)
% Convert from one surface to another
%
% srcsubj = 'fsaverage'
% trgsubj = 'KNK_S017'
% surf    = '/mnt/diskArray/archives/VOF/templateROIs/lh.VWFA1.label'
% outfile = 'tmp.label'
% hemi    = 'lh'
% cmd = fs_surf2surf(srcsubj, trgsubj, surf, outfile, hemi)

if ~isempty(strfind(surf,'.label'))
    cmd = sprintf('mri_surf2surf --hemi %s --srcsubject %s --trgsubject %s --sval-annot %s --tval %s',...
        hemi, srcsubj, trgsubj, surf, outfile);
else
    cmd = sprintf('mri_surf2surf --hemi %s --srcsubject %s --trgsubject %s --sval %s --tval %s',...
        hemi, srcsubj, trgsubj, surf, outfile);
end
system(cmd)