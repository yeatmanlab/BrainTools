function response = AnalyzeBehaviorWordTask(tabPath)
%%
tabPath = '/mnt/diskArray/projects/MEG/nlr/bdata/202_DD_2015-08-27 14_50_00.573000.tab';

if exist('tabPath','var') && ~isempty(tabPath)
    if exist(tabPath,'dir')
        d = dir(fullfile(tabPath,'*.tab'));
        for ii = 1:length(d)
            fid(ii) = fopen(fullfile(tabPath,d(ii).name));
        end
    else
        fid = fopen(tabPath);
    end
end
for f = 1:length(fid)
    c = textscan(fid(f),'%f%s%s','Headerlines',2)
    fclose(fid(f));
    str = c{2};
    ts  = c{1};
    val = c{3};
    %[num,txt,raw]=xlsread('/home/jyeatman/git/wmdevo/megtools/data/_2015-03-03 16_47_12.637379.xlsx');
    %raw=raw(3:end,:);
    kidx = find(strcmp('keypress',str));
    ktimes = ts(kidx);
    
    names = {'word_c254_p20', 'word_c254_p50','word_c137_p20','word_c141_p20', 'word_c254_p80' ,'rand_c254_p20', 'rand_c254_p50','rand_c137_p20','rand_c141_p20'}
    
    
    imtypes = 1:length(names);
    for ii = imtypes
        idx = find(strcmp(num2str(ii),val) & strcmp('imtype',str));
        % add 1 because we want the time of the flip which is the next row
        idx = idx+1;
        times = ts(idx);
        timetopress = [];
        for jj = 1:length(times)
            tmp = ktimes-times(jj);
            p = tmp(tmp>.2 & tmp<1.3);
            if ~isempty(p)
                timetopress(end+1) = p;
            end
        end
        response(f).(names{ii}) = timetopress;
    end
end
