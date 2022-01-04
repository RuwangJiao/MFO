function Population = EnvironmentalSelection(Population, N, RefPoint, nSample, epsn)
% The environmental selection of MFO-HypE

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons - epsn, N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo == MaxFNo);
    Choose = true(1, length(Last));
    while sum(Choose) > N-sum(Next)
        drawnow();
        Remain   = find(Choose);
        F        = CalHV(Population(Last(Remain)).objs, RefPoint, sum(Choose) - N + sum(Next), nSample);
        [~, del] = min(F);
        Choose(Remain(del)) = false;
    end
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
   
    %%% Sort the population for mating selection
    CV = sum(max(0, Population.cons - epsn), 2);
    if size(Population(CV == 0), 2) > 0
        %sort the feasible solutions
        FeasiblePop = Population(CV == 0);
        Rank        = -CalHV(FeasiblePop.objs, RefPoint, size(FeasiblePop, 2), nSample);
        [~, index2] = sort(Rank);
        FeasiblePop = FeasiblePop(index2(1:size(FeasiblePop, 2)));
    else
        FeasiblePop = [];
    end
    if size(Population(CV ~= 0), 2) > 0
        % sort the infeasible solutions
        InfeasiblePop = Population(CV ~= 0);
        [~, rank]     = sort(sum(max(0, InfeasiblePop.cons - epsn), 2));
        InfeasiblePop = InfeasiblePop(rank(1:size(InfeasiblePop, 2)));
    else
        InfeasiblePop = [];
    end
    Population = [FeasiblePop, InfeasiblePop];
end