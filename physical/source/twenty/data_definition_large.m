%% This file is a definition file used to define the datasets used internally
%definite k to get the real-time response
function Parameter = data_definition_large(k,Num_List)
    Parameter=Creating(k,Num_List);
    Parameter.w_number=Num_List(1);
    Parameter.p_number=Num_List(2);
    Parameter.tg_number=Num_List(3);
    Parameter.ess_number=Num_List(4);
    Parameter.dr_number=Num_List(5);
    
    % Uncertain Generation
    Parameter.Wind_f=Parameter.Wind_f(:,:,1:Parameter.w_number);
    Parameter.Solar_f=Parameter.Solar_f(:,:,1:Parameter.p_number);
    Parameter.Load_f=Parameter.Load_f(:,:,1:Parameter.dr_number);
    Parameter.Load_base=Parameter.Load_base(:,1:Parameter.dr_number);
    %% tg constraints
    Parameter.ag=[350,80,100];%%currency/(MW*15min)
    Parameter.bg=[0,20,80];%%currency/(15min)
    Parameter.cg=[300,70,90];%%currency/(MW*15min)^2
    Parameter.Pgmin=[0.1,0.1,0.1];%
    Parameter.Pgmax=[0.6,8,5];%output upper limit
    Parameter.Ug=[0.1,2.0,2.0];
    Parameter.Dg=[0.1,2.0,2.0];
    Parameter.ag= Parameter.ag(1:Parameter.tg_number);
    Parameter.bg= Parameter.bg(1:Parameter.tg_number);
    Parameter.cg= Parameter.cg(1:Parameter.tg_number);
    Parameter.Pgmin= Parameter.Pgmin(1:Parameter.tg_number);
    Parameter.Pgmax= Parameter.Pgmax(1:Parameter.tg_number);
    Parameter.Ug= Parameter.Ug(1:Parameter.tg_number);
    Parameter.Dg= Parameter.Dg(1:Parameter.tg_number);


    %% Energy storage case ESS1
    Parameter.as=zeros(Parameter.ess_number,1);%operating cost
    Parameter.yt_sout=[0.95,0.99,0.95,0.95,0.98,0.95,0.99,0.95,0.95,0.98,0.95,0.99,0.95,0.95,0.98];
    Parameter.yt_sin=[0.95,0.99,0.95,0.95,0.98,0.95,0.99,0.95,0.95,0.98,0.95,0.99,0.95,0.95,0.98,];%energy consumption constant
    Parameter.p_cmax=[1.5,1.5,1.8,2.0,2.5,1.5,1.5,1.8,2.0,2.5,1.5,1.5,1.8,2.0,2.5];%charging power
    Parameter.p_dmax=[1.5,1.5,1.8,2.0,2.5,1.5,1.5,1.8,2.0,2.5,1.5,1.5,1.8,2.0,2.5];%discharging power
    Parameter.socmin=[0.05,0.05,0.1,0.1,0.08,0.05,0.05,0.1,0.1,0.08,0.05,0.05,0.1,0.1,0.08];%minimum state of charge
    Parameter.socmax=[0.95,0.95,0.98,0.98,0.99,0.95,0.95,0.98,0.98,0.99,0.95,0.95,0.98,0.98,0.99];%maximum state of charge
    Parameter.Esmax=[8,8,10,10,12,8,8,10,10,12,8,8,10,10,12];%MW*15min
    Parameter.socstart=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5];
    Parameter.as= Parameter.as(1:Parameter.ess_number);
    Parameter.yt_sout= Parameter.yt_sout(1:Parameter.ess_number);
    Parameter.yt_sin= Parameter.yt_sin(1:Parameter.ess_number);
    Parameter.p_cmax= Parameter.p_cmax(1:Parameter.ess_number);
    Parameter.p_dmax= Parameter.p_dmax(1:Parameter.ess_number);
    Parameter.socmin= Parameter.socmin(1:Parameter.ess_number);
    Parameter.socmax= Parameter.socmax(1:Parameter.ess_number);
    Parameter.Esmax= Parameter.Esmax(1:Parameter.ess_number);
    Parameter.socstart= Parameter.socstart(1:Parameter.ess_number);

    %% cutting
    Parameter.Cutting=[685.00,651.45,667.20,700.56,689.86,633.36,685.50,613.38,689.13,695.31,695.09,...
        691.04,675.74,620.38,662.64];%currency/MW*15min
    Parameter.Cutting=Parameter.Cutting(1:Parameter.dr_number);
    
    %% Penalty level
    Parameter.pane=500;%penalty level currency/MW*15min
    Parameter.res=0;
end

