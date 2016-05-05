function [p,anovatab,stats] = HCP_anova1(x,group,displayopt,extra)
%   Hack of the built-in anova1 function to process HCP data analyzed with
%   AFQ with behavioral data loaded into the structure.

%ANOVA1 One-way analysis of variance (ANOVA).
%   ANOVA1 performs a one-way ANOVA for comparing the means of two or more 
%   groups of data. It returns the p-value for the null hypothesis that the
%   means of the groups are equal.
%
%   P = ANOVA1(M) for a matrix M treats each column as a separate group,
%   and determines whether the population means of the columns are equal.
%   This form of ANOVA1 is appropriate when each group has the same number
%   of elements (balanced ANOVA).
%
%   P = ANOVA1(V,GROUP) groups elements in the vector V according to values
%   in the grouping variable GROUP. GROUP must be a categorical variable,
%   numeric vector, logical vector, string array, or cell array of strings
%   with one group name for each element of X.  X values corresponding to
%   the same value of GROUP are placed in the same group.
%
%   P = ANOVA1(M,GROUP) accepts a character array or cell array of strings,
%   with one group name for each column of M. Columns with the same group
%   name are treated as part of the same group.
%
%   P = ANOVA1(X,GROUP,DISPLAYOPT) controls the display. DISPLAYOPT can be
%   'on' (the default) to display figures containing a standard one-way
%   anova table and a boxplot, or 'off' to omit these displays.  Note that
%   the notches in the boxplot provide a test of group medians (see HELP
%   BOXPLOT), and this is not the same as the F test for different means in
%   the anova table. X can be either a vector or matrix. If X is a matrix
%   and there are no group names, specify GROUP as [].
%
%   [P,ANOVATAB] = ANOVA1(...) returns the ANOVA table values as the
%   cell array ANOVATAB.
%
%   [P,ANOVATAB,STATS] = ANOVA1(...) returns an additional structure
%   of statistics useful for performing a multiple comparison of means
%   with the MULTCOMPARE function.
%
%   See also ANOVA2, ANOVAN, BOXPLOT, MANOVA1, MULTCOMPARE.

%   Reference: Robert V. Hogg, and Johannes Ledolter, Engineering Statistics
%   Macmillan 1987 pp. 205-206.

%   Copyright 1993-2013 The MathWorks, Inc.

% narginchk(1,4);

classical = 1;
% nargs = nargin;
% if (nargin>0 && strcmp(x,'kruskalwallis'))
%    % Called via kruskalwallis function, adjust inputs
%    classical = 0;
%    if (nargin >= 2), x = group; group = []; end
%    if (nargin >= 3), group = displayopt; displayopt = []; end
%    if (nargin >= 4), displayopt = extra; end
%    nargs = nargs-1;
% end
% 
% if (nargs < 2), group = []; end
% if (nargs < 3), displayopt = 'on'; end
% % Note: for backwards compatibility, accept 'nodisplay' for 'off'
% willdisplay = ~(strcmp(displayopt,'nodisplay') | strcmp(displayopt,'n') ...
%                 | strcmp(displayopt,'off'));

% Convert group to cell array from character array, make it a column
if (ischar(group) && ~isempty(group))
    group = cellstr(group);
end
if (size(group, 1) == 1)
    group = group';
end

% If the input is a matrix, it may not be balanced if it contains NaNs or
% if there are repeated grouping values, so turn it into a vector.
needvector = false;
if ~isvector(x)
    if (any(isnan(x(:))))
        needvector = true;
    elseif ~isempty(group) && length(group)~=length(unique(group))
        needvector = true;
    end
end

% If X is a matrix with NaNs, convert to vector form.
if ~isvector(x)
   if needvector
      [n,m] = size(x);
      x = x(:);
      gi = reshape(repmat((1:m), n, 1), n*m, 1);
      if isempty(group)     % no group names
         group = gi;
      elseif (size(group,1) == m)
         group = group(gi,:);
      else
         error(message('stats:anova1:InputSizeMismatch'));
      end
   end
elseif ~isempty(group) && (size(group,1) ~= length(x))
   error(message('stats:anova1:InputSizeMismatch'));
end

% If X is a matrix GROUP is provided with the correct size, use GROUP
% to define groups and to label boxes
if (~isempty(group) && (length(x) < numel(x)) ...
                    && (size(x,2) == size(group,1)))
   named = 1;
   [gid,gnames] = grp2idx(group);
   gnames = gnames(gid);
   grouped = 0;
else
   named = 0;
   gnames = [];
   grouped = ~isempty(group);
end

if (grouped)
   % Single data vector and a separate grouping variable
   x = x(:);
   lx = length(x);
   if (lx ~= numel(x))
      error(message('stats:anova1:VectorRequired'))
   end
   nonan = ~isnan(x);
   x = x(nonan);

   % Convert group to indices 1,...,g and separate names
   group = group(nonan,:);
   [groupnum, gnames] = grp2idx(group);
   named = 1;

   % Remove NaN values
   nonan = ~isnan(groupnum);
   if (~all(nonan))
      groupnum = groupnum(nonan);
      x = x(nonan);
   end

   lx = length(x);
   xorig = x;                    % use uncentered version to make M
   groupnum = groupnum(:);
   maxi = size(gnames, 1);
   if isa(x,'single')
      xm = zeros(1,maxi,'single');
   else
      xm = zeros(1,maxi);
   end
   countx = xm;
   if (classical)
      mu = mean(x);
      x = x - mu;                % center to improve accuracy
      xr = x;
   else
      [xr,tieadj] = tiedrank(x);
   end
   
   for j = 1:maxi
      % Get group sizes and means
      k = find(groupnum == j);
      lk = length(k);
      countx(j) = lk;
      xm(j) = mean(xr(k));       % column means
   end

   gm = mean(xr);                      % grand mean
   df1 = sum(countx>0) - 1;            % Column degrees of freedom
   df2 = lx - df1 - 1;                 % Error degrees of freedom
   xc = xm - gm;                       % centered
   xc(countx==0) = 0;
   RSS = dot(countx, xc.^2);           % Regression Sum of Squares
else
   % Data in matrix form, no separate grouping variable
   [r,c] = size(x);
   lx = r * c;
   if (classical)
      xr = x;
      mu = mean(xr(:));
      xr = xr - mu;           % center to improve accuracy
   else
      [xr,tieadj] = tiedrank(x(:));
      xr = reshape(xr, size(x));
   end
   countx = repmat(r, 1, c);
   xorig = x;                 % save uncentered version for boxplot
   xm = mean(xr);             % column means
   gm = mean(xm);             % grand mean
   df1 = c-1;                 % Column degrees of freedom
   df2 = c*(r-1);             % Error degrees of freedom
   RSS = r*(xm - gm)*(xm-gm)';        % Regression Sum of Squares
end

TSS = (xr(:) - gm)'*(xr(:) - gm);  % Total Sum of Squares
SSE = TSS - RSS;                   % Error Sum of Squares

if (df2 > 0)
   mse = SSE/df2;
else
   mse = NaN;
end

if (classical)
   if (SSE~=0)
      F = (RSS/df1) / mse;
      p = fpval(F,df1,df2);        % Probability of F given equal means.
   elseif (RSS==0)                 % Constant Matrix case.
      F = NaN;
      p = NaN;
   else                            % Perfect fit case.
      F = Inf;
      p = 0;
   end
else
   F = (12 * RSS) / (lx * (lx+1));
   if (tieadj > 0)
      F = F / (1 - 2 * tieadj/(lx^3-lx));
   end
   p = chi2pval(F,df1);
end


Table=zeros(3,5);               %Formatting for ANOVA Table printout
Table(:,1)=[ RSS SSE TSS]';
Table(:,2)=[df1 df2 df1+df2]';
Table(:,3)=[ RSS/df1 mse Inf ]';
Table(:,4)=[ F Inf Inf ]';
Table(:,5)=[ p Inf Inf ]';

colheads = {getString(message('stats:anova1:ColHeadSource')), getString(message('stats:anova1:ColHeadSS')), getString(message('stats:anova1:ColHeadDf')), getString(message('stats:anova1:ColHeadMS')), getString(message('stats:anova1:ColHeadF')), getString(message('stats:anova1:ColHeadProbGtF'))};

if (~classical)
   colheads{5} = getString(message('stats:anova1:ColHeadChisq'));
   colheads{6} = getString(message('stats:anova1:ColHeadProbGtChisq'));
end
rowheads = {getString(message('stats:anova1:RowHeadColumns')), getString(message('stats:anova1:RowHeadError')), getString(message('stats:anova1:RowHeadTotal'))};
if (grouped)
   rowheads{1} = getString(message('stats:anova1:RowHeadGroups'));
end

% Create cell array version of table
atab = num2cell(Table);
for i=1:size(atab,1)
   for j=1:size(atab,2)
      if (isinf(atab{i,j}))
         atab{i,j} = [];
      end
   end
end
atab = [rowheads' atab];
atab = [colheads; atab];
if (nargout > 1)
   anovatab = atab;
end

% Create output stats structure if requested, used by MULTCOMPARE
if (nargout > 2)
   if ~isempty(gnames)
      stats.gnames = gnames;
   else
      stats.gnames = strjust(num2str((1:length(xm))'),'left');
   end
   stats.n = countx;
   if (classical)
      stats.source = 'anova1';
      stats.means = xm + mu;
      stats.df = df2;
      stats.s = sqrt(mse);
   else
      stats.source = 'kruskalwallis';
      stats.meanranks = xm;
      stats.sumt = 2 * tieadj;
   end
end

if (~willdisplay), return; end

digits = [-1 -1 0 -1 2 4];
if (classical)
   wtitle = getString(message('stats:anova1:OnewayANOVA'));
   ttitle = getString(message(strcat('stats:anova1:ANOVATable', fgnames));
else
   wtitle = getString(message('stats:anova1:KruskalWallisOnewayANOVA'));
   ttitle = getString(message('stats:anova1:KruskalWallisANOVATable'));
end
tblfig = statdisptable(atab, wtitle, ttitle, '', digits);
set(tblfig,'tag','table');

f1 = figure('pos',get(gcf,'pos') + [0,-200,0,0],'tag','boxplot');
subplot(4, 5, fgnum); hold on;
ax = axes('Parent',f1);
if (~grouped)
   boxplot(ax,xorig,'notch','on');
else
   boxplot(ax,xorig,groupnum,'notch','on');
   h = get(ax,'XLabel');
   set(h,'String',getString(message('stats:anova1:GroupNumber')));
   title('fgnames');
end

% If there are group names, use them
if ~isempty(gnames)
   h = get(ax,'XLabel');
   if (named)
      set(h,'String','');
   end
   set(ax, 'xtick', (1:size(gnames,1)), 'xticklabel', gnames);
   title('fgnames');
end



%%
%     [fap, faT, faSt] = HCP_anova1(x,group,displayopt,extra)
    %   Hack of the built-in anova1 function to process HCP data analyzed with
    %   AFQ with behavioral data loaded into the structure.
    
    %ANOVA1 One-way analysis of variance (ANOVA).
    %   ANOVA1 performs a one-way ANOVA for comparing the means of two or more
    %   groups of data. It returns the p-value for the null hypothesis that the
    %   means of the groups are equal.
    %
    %   P = ANOVA1(M) for a matrix M treats each column as a separate group,
    %   and determines whether the population means of the columns are equal.
    %   This form of ANOVA1 is appropriate when each group has the same number
    %   of elements (balanced ANOVA).
    %
    %   P = ANOVA1(V,GROUP) groups elements in the vector V according to values
    %   in the grouping variable GROUP. GROUP must be a categorical variable,
    %   numeric vector, logical vector, string array, or cell array of strings
    %   with one group name for each element of X.  X values corresponding to
    %   the same value of GROUP are placed in the same group.
    %
    %   P = ANOVA1(M,GROUP) accepts a character array or cell array of strings,
    %   with one group name for each column of M. Columns with the same group
    %   name are treated as part of the same group.
    %
    %   P = ANOVA1(X,GROUP,DISPLAYOPT) controls the display. DISPLAYOPT can be
    %   'on' (the default) to display figures containing a standard one-way
    %   anova table and a boxplot, or 'off' to omit these displays.  Note that
    %   the notches in the boxplot provide a test of group medians (see HELP
    %   BOXPLOT), and this is not the same as the F test for different means in
    %   the anova table. X can be either a vector or matrix. If X is a matrix
    %   and there are no group names, specify GROUP as [].
    %
    %   [P,ANOVATAB] = ANOVA1(...) returns the ANOVA table values as the
    %   cell array ANOVATAB.
    %
    %   [P,ANOVATAB,STATS] = ANOVA1(...) returns an additional structure
    %   of statistics useful for performing a multiple comparison of means
    %   with the MULTCOMPARE function.
    %
    %   See also ANOVA2, ANOVAN, BOXPLOT, MANOVA1, MULTCOMPARE.
    
    %   Reference: Robert V. Hogg, and Johannes Ledolter, Engineering Statistics
    %   Macmillan 1987 pp. 205-206.
    
    %   Copyright 1993-2013 The MathWorks, Inc.
    
    % narginchk(1,4);
    
    classical = 1;
    % nargs = nargin;
    % if (nargin>0 && strcmp(x,'kruskalwallis'))
    %    % Called via kruskalwallis function, adjust inputs
    %    classical = 0;
    %    if (nargin >= 2), x = group; group = []; end
    %    if (nargin >= 3), group = displayopt; displayopt = []; end
    %    if (nargin >= 4), displayopt = extra; end
    %    nargs = nargs-1;
    % end
    %
    % if (nargs < 2), group = []; end
    % if (nargs < 3), displayopt = 'on'; end
    % % Note: for backwards compatibility, accept 'nodisplay' for 'off'
    % willdisplay = ~(strcmp(displayopt,'nodisplay') | strcmp(displayopt,'n') ...
    %                 | strcmp(displayopt,'off'));
    
    % Convert group to cell array from character array, make it a column
    if (ischar(group) && ~isempty(group))
        group = cellstr(group);
    end
    if (size(group, 1) == 1)
        group = group';
    end
    
    % If the input is a matrix, it may not be balanced if it contains NaNs or
    % if there are repeated grouping values, so turn it into a vector.
    needvector = false;
    if ~isvector(x)
        if (any(isnan(x(:))))
            needvector = true;
        elseif ~isempty(group) && length(group)~=length(unique(group))
            needvector = true;
        end
    end
    
    % If X is a matrix with NaNs, convert to vector form.
    if ~isvector(x)
        if needvector
            [n,m] = size(x);
            x = x(:);
            gi = reshape(repmat((1:m), n, 1), n*m, 1);
            if isempty(group)     % no group names
                group = gi;
            elseif (size(group,1) == m)
                group = group(gi,:);
            else
                error(message('stats:anova1:InputSizeMismatch'));
            end
        end
    elseif ~isempty(group) && (size(group,1) ~= length(x))
        error(message('stats:anova1:InputSizeMismatch'));
    end
    
    % If X is a matrix GROUP is provided with the correct size, use GROUP
    % to define groups and to label boxes
    if (~isempty(group) && (length(x) < numel(x)) ...
            && (size(x,2) == size(group,1)))
        named = 1;
        [gid,gnames] = grp2idx(group);
        gnames = gnames(gid);
        grouped = 0;
    else
        named = 0;
        gnames = [];
        grouped = ~isempty(group);
    end
    
    if (grouped)
        % Single data vector and a separate grouping variable
        x = x(:);
        lx = length(x);
        if (lx ~= numel(x))
            error(message('stats:anova1:VectorRequired'))
        end
        nonan = ~isnan(x);
        x = x(nonan);
        
        % Convert group to indices 1,...,g and separate names
        group = group(nonan,:);
        [groupnum, gnames] = grp2idx(group);
        named = 1;
        
        % Remove NaN values
        nonan = ~isnan(groupnum);
        if (~all(nonan))
            groupnum = groupnum(nonan);
            x = x(nonan);
        end
        
        lx = length(x);
        xorig = x;                    % use uncentered version to make M
        groupnum = groupnum(:);
        maxi = size(gnames, 1);
        if isa(x,'single')
            xm = zeros(1,maxi,'single');
        else
            xm = zeros(1,maxi);
        end
        countx = xm;
        if (classical)
            mu = mean(x);
            x = x - mu;                % center to improve accuracy
            xr = x;
        else
            [xr,tieadj] = tiedrank(x);
        end
        
        for j = 1:maxi
            % Get group sizes and means
            k = find(groupnum == j);
            lk = length(k);
            countx(j) = lk;
            xm(j) = mean(xr(k));       % column means
        end
        
        gm = mean(xr);                      % grand mean
        df1 = sum(countx>0) - 1;            % Column degrees of freedom
        df2 = lx - df1 - 1;                 % Error degrees of freedom
        xc = xm - gm;                       % centered
        xc(countx==0) = 0;
        RSS = dot(countx, xc.^2);           % Regression Sum of Squares
    else
        % Data in matrix form, no separate grouping variable
        [r,c] = size(x);
        lx = r * c;
        if (classical)
            xr = x;
            mu = mean(xr(:));
            xr = xr - mu;           % center to improve accuracy
        else
            [xr,tieadj] = tiedrank(x(:));
            xr = reshape(xr, size(x));
        end
        countx = repmat(r, 1, c);
        xorig = x;                 % save uncentered version for boxplot
        xm = mean(xr);             % column means
        gm = mean(xm);             % grand mean
        df1 = c-1;                 % Column degrees of freedom
        df2 = c*(r-1);             % Error degrees of freedom
        RSS = r*(xm - gm)*(xm-gm)';        % Regression Sum of Squares
    end
    
    TSS = (xr(:) - gm)'*(xr(:) - gm);  % Total Sum of Squares
    SSE = TSS - RSS;                   % Error Sum of Squares
    
    if (df2 > 0)
        mse = SSE/df2;
    else
        mse = NaN;
    end
    
    if (classical)
        if (SSE~=0)
            F = (RSS/df1) / mse;
            xunder = 1./max(0,F); % HAX
            xunder(isnan(F)) = NaN; %
            p = fcdf(xunder,df2,df1); %
%             p = fpval(F,df1,df2);        % Probability of F given equal means.
        elseif (RSS==0)                 % Constant Matrix case.
            F = NaN;
            p = NaN;
        else                            % Perfect fit case.
            F = Inf;
            p = 0;
        end
    else
        F = (12 * RSS) / (lx * (lx+1));
        if (tieadj > 0)
            F = F / (1 - 2 * tieadj/(lx^3-lx));
        end
        p = chi2pval(F,df1);
    end
    
    
    Table=zeros(3,5);               %Formatting for ANOVA Table printout
    Table(:,1)=[ RSS SSE TSS]';
    Table(:,2)=[df1 df2 df1+df2]';
    Table(:,3)=[ RSS/df1 mse Inf ]';
    Table(:,4)=[ F Inf Inf ]';
    Table(:,5)=[ p Inf Inf ]';
    
    colheads = {getString(message('stats:anova1:ColHeadSource')), getString(message('stats:anova1:ColHeadSS')), getString(message('stats:anova1:ColHeadDf')), getString(message('stats:anova1:ColHeadMS')), getString(message('stats:anova1:ColHeadF')), getString(message('stats:anova1:ColHeadProbGtF'))};
    
    if (~classical)
        colheads{5} = getString(message('stats:anova1:ColHeadChisq'));
        colheads{6} = getString(message('stats:anova1:ColHeadProbGtChisq'));
    end
    rowheads = {getString(message('stats:anova1:RowHeadColumns')), getString(message('stats:anova1:RowHeadError')), getString(message('stats:anova1:RowHeadTotal'))};
    if (grouped)
        rowheads{1} = getString(message('stats:anova1:RowHeadGroups'));
    end
    
    % Create cell array version of table
    atab = num2cell(Table);
    for i=1:size(atab,1)
        for j=1:size(atab,2)
            if (isinf(atab{i,j}))
                atab{i,j} = [];
            end
        end
    end
    atab = [rowheads' atab];
    atab = [colheads; atab];
%     if (nargout > 1)
        anovatab = atab;
%     end
    
    % Create output stats structure if requested, used by MULTCOMPARE
    if (nargout > 2)
        if ~isempty(gnames)
            stats.gnames = gnames;
        else
            stats.gnames = strjust(num2str((1:length(xm))'),'left');
        end
        stats.n = countx;
        if (classical)
            stats.source = 'anova1';
            stats.means = xm + mu;
            stats.df = df2;
            stats.s = sqrt(mse);
        else
            stats.source = 'kruskalwallis';
            stats.meanranks = xm;
            stats.sumt = 2 * tieadj;
        end
    end
    
%     if (~willdisplay), return; end
    
    digits = [-1 -1 0 -1 2 4];
    if (classical)
        wtitle = getString(message('stats:anova1:OnewayANOVA'));
        ttitle = strcat(getString(message('stats:anova1:ANOVATable')), fgnames);
    else
        wtitle = getString(message('stats:anova1:KruskalWallisOnewayANOVA'));
        ttitle = getString(message('stats:anova1:KruskalWallisANOVATable'));
    end
    tblfig = statdisptable(atab, wtitle, ttitle, '', digits);
    set(tblfig,'tag','table');
    
    f1 = figure('pos',get(gcf,'pos') + [0,-200,0,0],'tag','boxplot');
    subplot(4, 5, fgnum); hold on;
    ax = axes('Parent',f1);
    if (~grouped)
        boxplot(ax,xorig,'notch','on');
    else
        boxplot(ax,xorig,groupnum,'notch','on');
        h = get(ax,'XLabel');
        set(h,'String',getString(message('stats:anova1:GroupNumber')));
        title(fgnames);
    end
    
    % If there are group names, use them
    if ~isempty(gnames)
        h = get(ax,'XLabel');
        if (named)
            set(h,'String','');
        end
        set(ax, 'xtick', (1:size(gnames,1)), 'xticklabel', gnames);
        title(fgnames);
    end
    
    fap{fgnum} = p; 
    faT{fgnum} = anovatab;
    