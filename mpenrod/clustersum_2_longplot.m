% Function to make the control vs. intervention graphs for vertices of
% interest. Plots produced are averages of cortical thickness at the given
% timepoint with error bars noting the standard error of the mean
%
% Inputs:
% cont_TPs: A vertex containing the list of control subjects, including a marker
% ('_1','_2',etc.) noting the time point of the session
% inter_TPs: The same as cont_TPs, but for the intervention subjects
% hemi: the hemisphere represented in the data ('LH' or 'RH')
% vertices: A vertex containing all vertices of interest
% annots: A vertex containing the ROI which contains the vertex listed in
% the corresponding row in "vertices"
% inter_data: all data for the intervention subjects of interest 
% cont_data: all data for the control subjects of interest
%       ***NOTE*** This must have the same number of rows and be in the
%       same order as inter_TPs/cont_TPs. As long as the steps data preparation
% 	section of the LME FreeSurfer tutorial are followed and the 
%       inter_TPs/cont_TPs are sorted in ascending numerical order there
%       should be no problem 
% out_path: path to the desired folder for which files are saved as
% <hemi>_vertex<vertex #>_<annot @ vertex>.jpg
%
% Output:
% Plots depicting the average thickness at each time point at all vertices
% given. These figures allow visual representation of the path of CT change
% in the brains of intervention subjects and controls across timepoints.
% 
% Author: Mark Penrod
% Date: July 2017

function clustersum_2_longplot(cont_TPs,inter_TPs,hemi,vertices,annots,cont_data,inter_data,out_path)
% Identifying the rows in the various matrices which correspond to each
% timepoint such that the rows are organized in (a) timepoint matrix/matrices for
% which the column number corresponds to the timepoint
oo = 1;
tt = 1;
th = 1;
ff = 1;
for ii = 1:numel(cont_TPs)
    if ~isempty(strfind(cont_TPs{ii,1},'_1'))
        tp_cont_rows(oo,1) = ii;
        oo = oo + 1;
    elseif ~isempty(strfind(cont_TPs{ii,1},'_2'))
        tp_cont_rows(tt,2) = ii;
        tt = tt + 1;
    elseif ~isempty(strfind(cont_TPs{ii,1},'_3'))
        tp_cont_rows(th,3) = ii;
        th = th + 1;
    elseif ~isempty(strfind(cont_TPs{ii,1},'_4'))
        tp_cont_rows(ff,4) = ii;
        ff = ff + 1;
    end
end

oo = 1;
tt = 1;
th = 1;
ff = 1;
for ii = 1:numel(inter_TPs)
    if ~isempty(strfind(inter_TPs{ii,1},'_1'))
        tp_inter_rows(oo,1) = ii;
        oo = oo + 1;
    elseif ~isempty(strfind(inter_TPs{ii,1},'_2'))
        tp_inter_rows(tt,2) = ii;
        tt = tt + 1;
    elseif ~isempty(strfind(inter_TPs{ii,1},'_3'))
        tp_inter_rows(th,3) = ii;
        th = th + 1;
    elseif ~isempty(strfind(inter_TPs{ii,1},'_4'))
        tp_inter_rows(ff,4) = ii;
        ff = ff + 1;
    end
end

% Collect thickness values at vertices of interest for each time point and
% find the average and standard deviations for those thicknesses
% This is done for both control and intervention groups
for ii = 1:numel(vertices)
    vertex = vertices{ii,1};
    annot = annots{ii,1};
    for tp = 1:4
        for rr = 1:numel(tp_inter_rows(:,tp))
            if tp_inter_rows(rr,tp) ~= 0
                inter_th_at_vertex(rr,tp) = ...
                    inter_data(tp_inter_rows(rr,tp),vertex);
            end
        end
        inter_data_atTP = inter_th_at_vertex(find(inter_th_at_vertex(:,tp),1,'first'):find(inter_th_at_vertex(:,tp),1,'last'),tp);
        tp_inter_avgs(1,tp) = mean(inter_data_atTP);
        tp_inter_SEMs(1,tp) = (std(inter_data_atTP)/sqrt(length(inter_data_atTP)));
        for rr = 1:numel(tp_cont_rows(:,tp))
            if tp_cont_rows(rr,tp) ~= 0
                cont_th_at_vertex(rr,tp) = ...
                    cont_data(tp_cont_rows(rr,tp),vertex);
            end
        end
        cont_data_atTP = cont_th_at_vertex(find(cont_th_at_vertex(:,tp),1,'first'):find(cont_th_at_vertex(:,tp),1,'last'),tp);
        tp_cont_avgs(1,tp) = mean(cont_data_atTP);
        tp_cont_SEMs(1,tp) = (std(cont_data_atTP)/sqrt(length(cont_data_atTP)));
    end
    set(0,'DefaultFigureVisible','off');
    timepoints = [1,2,3,4];
    the_fig = figure; hold on
    errorbar(timepoints,tp_inter_avgs(1,:),tp_inter_SEMs(1,:),'b')
    errorbar(timepoints,tp_cont_avgs(1,:),tp_cont_SEMs(1,:),'g')
    legend('LowBRS','HighBRS','Location','southwest');
    box off
    xlabel('Timepoint')
    ylabel(strcat('Cortical Thickness at Vertex:',[' ',num2str(vertex)],' (',hemi,annot,')'))
    hold off
    
    
    saveas(the_fig,fullfile(out_path,strcat(hemi,'_vertex',num2str(vertex),'_',annot,'.jpg')))
    
    set(0,'DefaultFigureVisible','on');
end
