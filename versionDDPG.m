Ts=0.1 %time of one step
tf=20 %time of the simulation
obsInfo = rlNumericSpec([1 1],...
    'LowerLimit',[0]',...
    'UpperLimit',[8e6]'); %Pload range [0 to 8Mw] IT STANDS FOR OUR STATE
obsInfo.Name = 'observations';
numObservations = obsInfo.Dimension(1);
actInfo = rlNumericSpec([4 1],... % here i want to create an action with the value of each power sources
    'LowerLimit',[0 0 0 0]',...  % so i want to have an action like that A=[Pdg1,Pdg2,Pfc], o fid not put the Pbatt because 
   'UpperLimit',[3e6 3e6 1e6 1.5e6]'); % we will condider that Pbatt=Pload-Pdg1-Pdg2-Pfc
numActions = actInfo.Dimension(1); %here it will be 3


env = rlSimulinkEnv('testrl','testrl/RLAgent8',... %here i have to define for the code the simulink environnement, the first two argument are the name of the file and the path to the RL agent block
    obsInfo,actInfo);
% env.ResetFcn = @(in)localResetFcn(in);
rng(0)



statePath = [ %those lines are for the critic of our solutions, we create a layer which take the state and the action chosen and it is supposed to learn and approximate the best pattern 
    imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
    fullyConnectedLayer(1,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1') 
    fullyConnectedLayer(25,'Name','CriticStateFC2')]; % end of the state part
actionPath = [
    %reluLayer('Name','CriticRelu3')
    imageInputLayer([numActions 1 1],'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')]; %end of the action part
commonPath = [ % we combine them together 
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts); % the function which create the critic of our problem 
% actorPath = [ % the network is to choose the action when we give them a state
%     imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
%     fullyConnectedLayer(1, 'Name','actorFC')
% %   tanhLayer('Name','actorTanh')
%     fullyConnectedLayer(numActions,'Name','Action')
%     ];
% actorNetwork=layerGraph(actorPath)
% actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
% actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
actorPath = [ % the network is to choose the action when we give them a state
   imageInputLayer([numObservations 1 1],'Normalization','none','Name','State')
%    fullyConnectedLayer(2, 'Name','actorFC')
%    fullyConnectedLayer(2, 'Name','actorFC2')
   fullyConnectedLayer(numActions,'Name','Action','BiasLearnRateFactor',0,'Bias',[0;0;0;0])
   %tanhLayer('Name','actorTanh')
     ];
 actorNetwork=layerGraph(actorPath)
 actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
 actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
agentOpts = rlDDPGAgentOptions(... % in DPPG method an agent is the combinaison of (actor/critic layers)
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99); % i put this small discount factor to limit "the long-term reward", we want a blind algorithm i.e it doesnt have to think about the future to generate the next step.   

agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agent = rlDDPGAgent(actor,critic,agentOpts);
maxepisodes = 10000;
maxsteps=ceil(tf/Ts);
trainOpts = rlTrainingOptions(... % here it's the option of our training the commented options can permite to save agents with a good episode reward and put it in files to test them while the code is still running
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ... % nombre de fois par episode
    'ScoreAveragingWindowLength',5, ...cccc
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',100000000000000) 
%     'SaveAgentCriteria','EpisodeReward',...
%     'SaveAgentValue',-2,...
%     'SaveAgentDirectory','C:\Users\conqu\Desktop\dossierdesdeeplearning\agentsave1\psodeep');
     
trainingStats = train(agent,env,trainOpts);
opt = rlRepresentationOptions('UseDevice',"gpu"); % here maybe you computer can do it, mine can not, it is  possible to use the GPU of your computer to accelere the training

%  function in = localResetFcn(in) % this function permites to change the Pload value each time we do an other episode it permites to have a high range of testing
%  blk = sprintf('testrl/NOM');
%  
%  h=8e6*rand
%  in = setBlockParameter(in,blk,'Value',num2str(h));
% 
%  end





