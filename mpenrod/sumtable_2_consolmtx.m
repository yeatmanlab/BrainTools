

function [out_mtx,ROIs] = sumtable_2_consolmtx(tbl)
rr = 1;
ll = 1;
ROIs = {};
for ii = 2:numel(tbl.Properties.VariableNames)-2
    ROI = tbl.Properties.VariableNames(ii);
    if ~ismember(char(ROI),ROIs)
        ROIs(ll) = ROI;
        ll = ll + 1;
    end
    for jj = 1:rows(tbl)
        subject = tbl{jj,1};
        out_mtx(rr,1) = subject;
        out_mtx{rr,2} = tbl{jj,ii};
        if strfind(char(subject),'_1.')
            out_mtx{rr,3} = 1;
        elseif strfind(char(subject),'_2.')
            out_mtx{rr,3} = 2;
        elseif strfind(char(subject),'_3.')
            out_mtx{rr,3} = 3;
        elseif strfind(char(subject),'_4.')
            out_mtx{rr,3} = 4;
        end
        out_mtx(rr,4) = ROI;
        rr = rr + 1;
    end
end
end