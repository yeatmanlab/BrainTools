function [C,include,exclude] = nf_excludeMotion(sublist,session)

root_dir = '/mnt/disks/scratch/PREK_Analysis/data';

for ii = 1:length(sublist)
    denoisedDir = strcat(root_dir,'/',sublist{ii},'/',session,'/func/GLMdenoise');
    if exist(denoisedDir,'dir') == 7
        fig_dir = strcat(root_dir, '/', sublist{ii}, '/',session,'/func/figures');
        cd(fig_dir);
        load('record.mat')
        motion1 = max(sqrt(sum(diff(mparams{1}(:,1:3)).^2,2)));
        motion2 = max(sqrt(sum(diff(mparams{2}(:,1:3)).^2,2)));
        C{ii,1} = sublist{ii};
        C{ii,2} = max(motion1);%max([motion1 motion2]);
        C{ii,3} = max(motion2);
    else
      C{ii,1} = NaN;
      C{ii,2} = NaN;
      C{ii,3} = NaN; 
    end 
end

include = {};
run = {};
for ii = 1:length(sublist)
    if (C{ii,2} < 6) || (C{ii,3} < 6)
        include = [include; C(ii,1)];
    
        if C{ii,2} < C{ii,3} 
            run = [run;1];
        else
            run=[run;2];
        end 
    end 
end 
        
exclude = {};
for ii = 1:length(sublist)
    if (C{ii,2} > 6) && (C{ii,3} > 6)
        exclude = [exclude; C(ii,1)];
    end
end 

include = [include run];