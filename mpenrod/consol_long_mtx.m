%
% Function which takes a 3D matrix and organizes it into a single 2D
% matrix, necessary for use in lme_long_fitandplot.
%
% Input:
% long_mtx: the 3D matrix for which the first column is the subject ID, the
% next 4 are the session data, and the 6th is the ROI. Each layer of the
% matrix contains the data for a different ROI
%
% Outputs:
% consol_mtx: A matrix containing (in terms of columns) 
%   (1) The subject IDs
%   (2) The average cortical thickness for the ROI 
%   (3) The timepoint 
%   (4) The ROI
% ROIs: A list of all the ROIs 
%
% Author: Mark Penrod
% Date: July 2017


function [consol_mtx,ROIs] = consol_long_mtx(long_mtx)
mtx_size = size(long_mtx);
rr = 1;

for subj = 1:mtx_size(1)
    for ll = 1:mtx_size(3)
        for session = 1:4
            consol_mtx{rr,1} = long_mtx{subj,1,ll};
            if isempty(long_mtx(subj,session+1,ll))
                consol_mtx{rr,2} = NaN;
            else
                consol_mtx{rr,2} = long_mtx{subj,session+1,ll};
            end
            consol_mtx{rr,3} = session;
            consol_mtx{rr,4} = long_mtx{subj,mtx_size(2),ll};
            rr = rr + 1;
        end
        ROIs{ll} = long_mtx{subj,mtx_size(2),ll};
    end
end
end
