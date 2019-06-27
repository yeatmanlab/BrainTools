function [C,include,exclude] = nf_excludeMotion(sublist)

root_dir = '/mnt/scratch/PREK_Analysis';

for ii = 1:length(sublist)
    denoisedDir = strcat(root_dir,'/',sublist{ii},'/ses-pre/func/GLMdenoise');
    if exist(denoisedDir,'dir') == 7
        fig_dir = strcat(root_dir, '/', sublist{ii}, '/ses-pre/func/figures');
        cd(fig_dir);
        load('record.mat')
        motion1 = max(sqrt(sum(diff(mparams{1}(:,1:3)).^2,2)));
        motion2 = max(sqrt(sum(diff(mparams{2}(:,1:3)).^2,2)));
        C{ii,1} = sublist{ii};
        C{ii,2} = max([motion1 motion2]);
    end
end

include = {};
for ii = 1:length(sublist)
    if C{ii,2} < 6
        include = [include; C(ii,1)];
    end
end 
        
exclude = {};
for ii = 1:length(sublist)
    if C{ii,2} > 6
        exclude = [exclude; C(ii,1)];
    end
end 