classdef MFOMOEAD < ALGORITHM
% <multi> <real/binary/permutation> <constrained>
% % Multiform optimization framework on MOEA/D

%------------------------------- Reference --------------------------------
% A Multiform Optimization Framework for Constrained Multi-Objective Optimization
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm, Problem)
            %% Generate the weight vectors
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            T             = 20;
            nr            = 2;
            %% Detect the neighbours of each solution
            B      = pdist2(W, W);
            [~, B] = sort(B, 2);
            B      = B(:, 1:T);
            %% Generate populations for target and source tasks, respectively
            TargetPop = Problem.Initialization();
            SourcePop = TargetPop;
            nCon      = size(TargetPop.cons, 2);
            initialE  = max(max(0, TargetPop.cons), [], 1);
            initialE(initialE<1) = 1;

            %% Optimization
            while Algorithm.NotTerminated(TargetPop)
                epsn       = ReduceBoundary(initialE, ceil(Problem.FE/Problem.N), ceil(Problem.maxFE/Problem.N)-1);
                % For each solution
                for i = 1 : Problem.N     
                    % Choose the parents
                    if rand < 0.9
                        P = B(i, randperm(size(B, 2)));
                    else
                        P = randperm(Problem.N);
                    end
                    rnd = randi(T, 1, 2);
                    x_1 = rnd(1);
                    x_2 = rnd(2);
                    if rand < 0.5
                        P1 = SourcePop(P(x_1));
                    else
                        P1 = TargetPop(P(x_1));
                    end
                    if rand < 0.5
                        P2 = SourcePop(P(x_2));
                    else
                        P2 = TargetPop(P(x_2));
                    end
                    MatingPool = [P1, P2];
                    % Generate an offspring
                    Offspring  = OperatorGAhalf(MatingPool);

                    % Update the ideal point
                    ZT = min(min(TargetPop.objs, [], 1), Offspring.obj);
                    ZS = min(min(SourcePop.objs, [], 1), Offspring.obj);

                    % The constraint violation of offspring, SourcePop, TargetPop
                    CVSP = max(0, SourcePop(P).cons); 
                    CVTP = sum(max(0, TargetPop(P).cons), 2);
                    CVSO = max(0, Offspring.con);
                    CVTO = sum(max(0, Offspring.con));
            
                    OObj  = Offspring.obj;
                    normW = sqrt(sum(W(P, :).^2, 2));
                    % Update solutions in TargetPop by PBI approach
                    TPObj    = TargetPop(P).objs;
                    normTP   = sqrt(sum((TPObj - repmat(ZT, length(P), 1)).^2, 2));
                    normTO   = sqrt(sum((OObj - ZT).^2, 2));
                    CosineTP = sum((TPObj - repmat(ZT, length(P), 1)).*W(P, :), 2)./normW./normTP;
                    CosineTO = sum(repmat(OObj - ZT, length(P), 1).*W(P, :), 2)./normW./normTO;
                    g_TPold  = normTP.*CosineTP + 5*normTP.*sqrt(1 - CosineTP.^2);
                    g_TOnew  = normTO.*CosineTO + 5*normTO.*sqrt(1 - CosineTO.^2);
                    TargetPop(P(find(g_TPold>=g_TOnew & CVTP==CVTO | CVTP>CVTO, nr))) = Offspring;

                    % Update solutions in SourcePop by PBI approach
                    SPObj    = SourcePop(P).objs;
                    normSP   = sqrt(sum((SPObj - repmat(ZS, length(P), 1)).^2, 2));
                    normSO   = sqrt(sum((OObj - ZS).^2, 2));
                    CosineSP = sum((SPObj - repmat(ZS, length(P), 1)).*W(P, :), 2)./normW./normSP;
                    CosineSO = sum(repmat(OObj - ZS, length(P), 1).*W(P, :), 2)./normW./normSO;
                    g_SPold  = normSP.*CosineSP + 5*normSP.*sqrt(1 - CosineSP.^2);
                    g_SOnew  = normSO.*CosineSO + 5*normSO.*sqrt(1 - CosineSO.^2);
                    Po       = P(find((sum(CVSO<=epsn,2)==nCon&sum(CVSP<=epsn,2)==nCon&g_SPold>=g_SOnew) | (sum(CVSO<=epsn,2)==nCon&sum(CVSP<=epsn,2)<nCon) | (sum(CVSO<=epsn,2)<nCon&sum(CVSP<=epsn,2)<nCon&sum(max(0,SourcePop(P).cons),2)>sum(max(0,Offspring.con))), nr));
                    SourcePop(Po) = Offspring;
                end
            end
        end
    end
end

function epsn = ReduceBoundary(eF, k, MaxK)
    %% Reduce the epsilon constraint boundary for source task
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = MaxK./power(log((eF + z)./z), 1.0./10);
    B(B==0)  = B(B==0) + Nearzero;
    f        = eF.* exp( -(k./B).^10 );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    epsn     = f - z;
    epsn(epsn<=0) = 0;
end