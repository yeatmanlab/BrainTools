function X = makeDesignMatrixFromParfile(parfile, TR, block_trs)
 
[onsets, conds] = readParFile(parfile);
% remove zeros (fixation)
onsets = onsets(conds~=0);
conds = conds(conds~=0);

%create matrix of zeros with dimensions #TRs x nConditions
nTRs = 101; 
nConds = 3; %5 contrast levels x 2 positions + 1 zero contrast
X = zeros(nTRs, nConds);
 
%fill in design matrix with 1's indicating event onset and type (e.g. event
%at tr = 6 that was condition 4 would be indicated by a 1 in row 6, col 4)
for i = 1:length(onsets)
    cur_tr = (onsets(i)/TR)+1;
    cur_cond = conds(i);
    X(cur_tr:cur_tr+block_trs-1, cur_cond) = 1;
end
 
%check;
figure; imagesc(X);
colormap(gray); colorbar;
xlabel('Conditions');
ylabel('Time points');
save(parfile(1:end-4),'X')