Ts=1 %time step 
tf=20 %time of the simulation 
load nouveaubail4d.txt %observation file
load actinfocharp4d.txt %action file
a=actinfocharp4d(:,1:end)
v=nouveaubail4d(:,1:end)
vprime=v.'
    aprime=a.'
c=num2cell(vprime,1)
c2=num2cell(aprime,1)
obsInfo = rlFiniteSetSpec(c)
actInfo = rlFiniteSetSpec(c2)
%actInfo = rlFiniteSetSpec({[0;1],[0;2],[0;0],[1;1],[1;0],[1;2],[2;0],[2;1],[2;2],[3;0],[3;1],[3;2],[4;0],[4;1],[4;2]}.')
numObservations = obsInfo.Dimension(1) 
numActions = actInfo.Dimension(1);


env = rlSimulinkEnv('MODELE4DIM','MODELE4DIM/RLAgent8',...
    obsInfo,actInfo);
%env.ResetFcn = @(in)localResetFcn(in);
rng(0)

statePath = [ %standard layer 
   
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
    'MaxStepsPerEpisode',maxsteps, ... % nombre de fois par episode,  20 NOW TO BE FASTER 
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingValue',8e10,... 
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',1500,...
    'SaveAgentDirectory','C:\Users\conqu\Desktop\dossierdesdeeplearning\CHARPENTIER4d') %to save the agent 
%'StopOnError','off'); 
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',true,... 
    'TargetUpdateMethod',"periodic",...
    'SampleTime',Ts,...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e5,...
    'DiscountFactor',0.99); 
agentOpts.EpsilonGreedyExploration.Epsilon=0.999
agentOpts.EpsilonGreedyExploration.EpsilonMin=0.01
agentOpts.EpsilonGreedyExploration.EpsilonDecay=0.00000989
disp(agentOpts.EpsilonGreedyExploration.Epsilon)

agent = rlDQNAgent(critic,agentOpts);
trainingStats = train(agent,env,trainOpts);




