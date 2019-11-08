Ts=0.1 
tf=20
% load actiondiscret.txt
% v=actiondiscret(:,1:end)
% vprime=v.'
% c=num2cell(vprime,1)
% actInfo = rlFiniteSetSpec(c)
actInfo = rlFiniteSetSpec({[2.75;2.75;0.5;0]*1e6,[2;3;1;1]*1e6,[3;3;0;0]*1e6,[3;3;1;1]*1e6,[3;3;0;-1]*1e6,[3;3;1;-1]*1e6,[1;1;1;1]*1e6}.')
obsInfo = rlNumericSpec([1 1],...
    'LowerLimit',[0]',...
    'UpperLimit',[8e6]'); 
% load statediscret.txt
% w=statediscret(:,1)
% c2=num2cell(w)
% obsInfo=rlFiniteSetSpec(c2)
numObservations = obsInfo.Dimension(1) 
numActions = actInfo.Dimension(1);


env = rlSimulinkEnv('testrl','testrl/RLAgent8',...
    obsInfo,actInfo);
%env.ResetFcn = @(in)localResetFcn(in);
rng('shuffle')

statePath = [ % do not ask me why i put all those layer types i dont know, in mathwork exemples they use those
    %so for a first try i put those
   
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(1,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    %reluLayer('Name','CriticRelu3')
    imageInputLayer([numActions 1 1],'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% figure
% plot(criticNetwork)
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

maxepisodes = 100000;
maxsteps=ceil(tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ... % nombre de fois par episode
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingValue',8e10) %here i do not really know what to put i put 800 because it will not stop at all
%'StopOnError','off'); 
agentOpts = rlDQNAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',1e-10); 
agentOpts.EpsilonGreedyExploration.Epsilon=0.99
agentOpts.EpsilonGreedyExploration.EpsilonMin=0.001
agentOpts.EpsilonGreedyExploration.EpsilonDecay=0.000001
agent = rlDQNAgent(critic,agentOpts);
trainingStats = train(agent,env,trainOpts);


% function in = localResetFcn(in) % this function permites to change the Pload value each time we do an other episode 
% blk = sprintf('testrl/PLOADREF');
% h = round(rand*80+1);
% w=zeros(81,1)
% index1=1
% for i=0:80
%     w(index1,1)=0+0.1*i
%     index1=index1+1
% end
% entier=w(h)
% in = setBlockParameter(in,blk,'Value',num2str(entier));
% 
%  end

