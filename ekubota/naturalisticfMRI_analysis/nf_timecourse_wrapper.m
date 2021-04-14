hems = {'lh','rh'};

%rois = {'FG3','FG4','mFus','pFus','OTS','pOTS'};
%sessions = {'ses-pre','ses-post'};
rois = {'FG4'};
sessions = {'ses-post'};

for h = 1:2
    for r = 1:length(rois)
        for s = 1:length(sessions)
            roiName = [hems{h},'_',rois{r},'.nii.gz'];
            session = sessions{s};
            nf_mvpa_timecourse(roiName,session)
        end 
    end 
end 