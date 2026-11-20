function Parameter=Creating(k,VPP_list)
    creatingDir = fileparts(mfilename('fullpath'));
    if k==1
        Data=readtable(fullfile(creatingDir, 'Used_Data', 'Data_Scenario_Large.xlsx'));
    else
        Data=readtable(fullfile(creatingDir, 'Used_Data', 'Data_Scenario_Large_Extreme.xlsx'));
    end

    names=Data.Properties.VariableNames;
    wpp_cols=sort_member_cols(names,"wpp_");
    pvp_cols=sort_member_cols(names,"pvp_");
    dr_cols=sort_member_cols(names,"dr_");
    lmpda_col="LMPDA";
    if ~ismember(lmpda_col,names)
        error("LMPDA column not found in scenario data.");
    end

    if nargin < 2 || isempty(VPP_list)
        w_number=numel(wpp_cols);
        p_number=numel(pvp_cols);
        tg_number=0;
        ess_number=0;
        dr_number=numel(dr_cols);
    else
        w_number=VPP_list(1);
        p_number=VPP_list(2);
        tg_number=VPP_list(3);
        ess_number=VPP_list(4);
        dr_number=VPP_list(5);
    end

    if w_number > numel(wpp_cols) || p_number > numel(pvp_cols) || dr_number > numel(dr_cols)
        error("Not enough columns in scenario data for requested member counts.");
    end
    wpp_cols=wpp_cols(1:w_number);
    pvp_cols=pvp_cols(1:p_number);
    dr_cols=dr_cols(1:dr_number);

    Base=readtable(fullfile(creatingDir, 'Dataset', 'Dataset_Large_Case_Used_50.xlsx'));
    Wind_base=Base{:,wpp_cols};
    Solar_base=Base{:,pvp_cols};
    Load_base=Base{:,dr_cols};
    Load_base = str2double(string(Base{:,dr_cols}));
    Lmpda_base=Base{:,lmpda_col};

    [Wind_f,K]=read_member_matrix(Data,wpp_cols);
    Solar_f=read_member_matrix(Data,pvp_cols);
    Load_f=read_member_matrix(Data,dr_cols);
    Lmpda=parse_matrix(Data.(lmpda_col));
    P=parse_vector(Data.p);

    Parameter=struct;
    Parameter.Wind_base=Wind_base/1000;
    Parameter.Solar_base=Solar_base/1000;
    Parameter.Load_base=Load_base/1000;
    Parameter.P=P;

    Parameter.Lmpda=reload(Lmpda,max(Lmpda_base),min(Lmpda_base))/4;
    Parameter.Lmprt=reload(Lmpda,max(Lmpda_base),min(Lmpda_base))/4;

    wind_max=max(Wind_base,[],1);
    wind_min=min(Wind_base,[],1);
    solar_max=max(Solar_base,[],1);
    solar_min=min(Solar_base,[],1);
    load_max=max(Load_base,[],1);
    load_min=min(Load_base,[],1);

    Parameter.Wind_f=reload(Wind_f,reshape(wind_max,1,1,[]),reshape(wind_min,1,1,[]))/1000;
    Parameter.Solar_f=reload(Solar_f,reshape(solar_max,1,1,[]),reshape(solar_min,1,1,[]))/1000;
    Parameter.Load_f=reload(Load_f,reshape(load_max,1,1,[]),reshape(load_min,1,1,[]))/1000;

    function cols=sort_member_cols(list,prefix)
        mask=startsWith(list,prefix);
        cols=list(mask);
        cols=string(cols);
        nums=zeros(size(cols));
        for i=1:numel(cols)
            tail=extractAfter(cols(i),prefix);
            val=str2double(tail);
            if ~isnan(val)
                nums(i)=val;
            else
                nums(i)=inf;
            end
        end
        [~,order]=sort(nums);
        cols=cellstr(cols(order));
    end
    function [Data,K]=read_member_matrix(table_data,cols)
        first=parse_matrix(table_data.(cols{1}));
        K=size(first,2);
        Data=zeros(size(first,1),K,numel(cols));
        Data(:,:,1)=first;
        for c=2:numel(cols)
            Data(:,:,c)=parse_matrix(table_data.(cols{c}));
        end
    end
    function Data=parse_matrix(value)
        if iscell(value)
            value=value{1};
        end
        if isstring(value)
            value=char(value);
        end
        Data=jsondecode(value);
        Data=double(Data);
    end
    function Data=parse_vector(value)
        if iscell(value)
            value=value{1};
        end
        if isstring(value)
            value=char(value);
        end
        Data=jsondecode(value);
        Data=double(Data(:));
    end
    function Data=reload(data,Pmax,Pmin)
        Data=data.*(Pmax-Pmin)+Pmin;
    end
end
