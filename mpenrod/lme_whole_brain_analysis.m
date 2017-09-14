%
% Function which will take the output of long_prepare_LME and produce a
% significance map, calculating the change over time for each voxel in the
% brain
%
% Inputs:
% tp_Ms: a cell array of all the design matrices for each time point
% tp_Ys: a cell array of all the data matrices for each time point
% mri: the mri structure produced from the long_prepare_LME function where
% files from all time points were included
% variable_names: the column headers of the table, the same names should be
% used for the hypothesis
% hypothesis: a string that is the hypothesis being tested in the LME model
% out_file: the name of the output sig file
%
% Output:
% An mgh file which can be visually mapped onto the cortical surface or
% processed through mri_surfcluster in order to visualize or analyze,
% respectively, the lme data
%
% Author: Mark Penrod
% Date: August 3, 2017


function lme_whole_brain_analysis(tp_Ms,tp_Ys,mri,variable_names,hypothesis,out_file)
pval_map = zeros(cols(tp_Ys{1}),1);
for ii = 1:cols(tp_Ys{1})
    disp(strcat(num2str(ii),' out of', [' ',num2str(cols(tp_Ys{1}))]))
    samp = tp_Ys{1};
    if samp(:,ii) ~= 0
        first_M = tp_Ms{1};
        first_Y = tp_Ys{1};
        temp_mtx = [first_M(:,1),first_Y(:,ii),first_M(:,3)];
        for jj = 2:numel(tp_Ms)
            curr_M = tp_Ms{jj};
            curr_Y = tp_Ys{jj};
            temp_mtx = [temp_mtx(:,1),temp_mtx(:,2),temp_mtx(:,3);...
                curr_M(:,1),curr_Y(:,ii),curr_M(:,3)];
        end
        dtable = table(temp_mtx(:,1),temp_mtx(:,2),temp_mtx(:,3),'VariableNames',variable_names);
        lme = fitlme(dtable,hypothesis);
        pval_map(ii,1) = lme.anova.pValue(2);
    else
        pval_map(ii,1) = 1;
    end
end
mri.volsz = mri.volsz(1:2);
fs_write_Y(-log10(pval_map),mri,out_file);
end