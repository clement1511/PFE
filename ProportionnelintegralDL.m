Ts = 0.01;
Tf = 2;
obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -inf 0]',...
    'UpperLimit',[inf inf inf]'); %Pload range [0 to 8] 
obsInfo.Name = 'observations';
numObservations = obsInfo.Dimension(1);
%actInfo = rlNumericSpec([1 1],...
 %   'LowerLimit',[0]',...
  %  'UpperLimit',[3]');
%actInfo.Name = 'flow';
%numActions = actInfo.Dimension(1);
actInfo = rlNumericSpec([1 1]);
actInfo.Name = 'flow';
numActions = actInfo.Dimension(1);


env = rlSimulinkEnv('testDeep','testDeep/control/diesel/RLAgent',...
    obsInfo,actInfo);
% env.ResetFcn = @(in)localResetFcn(in);
rng(0)



statePath = [ % do not ask me why i put all those layer types i dont know, in mathwork exemples they use those
    %so for a first try i put those
   
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(1,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    
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

% agentOpts = rlDQNAgentOptions(...
%     'UseDoubleDQN',false, ...    
%     'TargetUpdateMethod',"periodic", ...
%     'TargetUpdateFrequency',4, ...   
%     'ExperienceBufferLength',100000, ...
%     'DiscountFactor',0.99, ...
%     'MiniBatchSize',256);
% agent = rlDQNAgent(critic,agentOpts);








actorPath = [
    
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(3, 'Name','actorFC')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];
actorNetwork=layerGraph(actorPath)
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agent = rlDDPGAgent(actor,critic,agentOpts);

maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ... % nombre de fois par episode
    'ScoreAveragingWindowLength',20, ...cccc
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10000,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',10000,...
    'SaveAgentDirectory','C:\Users\conqu\Desktop\dossierdesdeeplearning\agentsave1');
%     'UseParallel',1
%     'ParallelizationOptions','asinc'

    
    
    
 %here i do not really know what to put i put 800 because it will not stop at all
%'StopOnError','off'); 

trainingStats = train(agent,env,trainOpts);


% function in = localResetFcn(in) % this function permites to change the Pload value each time we do an other episode 
% blk = sprintf('testDeep/wref');
% h =rand;
% in = setBlockParameter(in,blk,'Value',num2str(h));
%  end
