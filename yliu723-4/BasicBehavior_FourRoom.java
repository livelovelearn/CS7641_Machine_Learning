package reinforceLearning;

import java.awt.Color;
import java.util.List;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.GoalBasedRF;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.OOMDPPlanner;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.commonpolicies.BoltzmannQPolicy;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.auxiliary.common.ConstantStateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.SinglePFTF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.visualizer.Visualizer;

public class BasicBehavior_FourRoom {

	GridWorldDomain gwdg;
	Domain domain;
	StateParser sp;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	DiscreteStateHashFactory hashingFactory;

	public static void main(String[] args) {

		BasicBehavior_FourRoom example = new BasicBehavior_FourRoom();
		String outputPath = "output/";

		// uncomment the example you want to see (and comment-out the rest)
		long startTime = System.currentTimeMillis();

		example.ValueIterationExample(outputPath);

		//example.QLearningExample(outputPath);

		//example.PolicyIterationExample(outputPath);
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Time: "+elapsedTime);

		// run the visualizer (only use if you don't use the experiment plotter
		// example)
		example.visualize(outputPath);

	}

	public BasicBehavior_FourRoom() {

		// create the domain
		gwdg = new GridWorldDomain(11, 11);
		gwdg.setMapToFourRooms();
		gwdg.setProbSucceedTransitionDynamics(0.8); // set to non-deterministic
													// transition
		domain = gwdg.generateDomain();

		// create the state parser
		sp = new GridWorldStateParser(domain);

		// define the task
		rf = new UniformCostRF();
		tf = new SinglePFTF(
				domain.getPropFunction(GridWorldDomain.PFATLOCATION));
		goalCondition = new TFGoalCondition(tf);

		// set up the initial state of the task
		initialState = GridWorldDomain.getOneAgentOneLocationState(domain);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, 10, 10);

		// set up the state hashing system
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory
				.setAttributesForClass(
						GridWorldDomain.CLASSAGENT,
						domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);

		// add visual observer
		VisualActionObserver observer = new VisualActionObserver(domain,
				GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		((SADomain) this.domain).setActionObserverForAllAction(observer);
		observer.initGUI();

	}

	public void visualize(String outputPath) {
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		EpisodeSequenceVisualizer evis = new EpisodeSequenceVisualizer(v,
				domain, sp, outputPath);
	}

	public void ValueIterationExample(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}

		OOMDPPlanner planner = new ValueIteration(domain, rf, tf, 0.99,
				hashingFactory, 0.001, 100);
		planner.planFromState(initialState);

		// create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner) planner);

		// record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(
				outputPath + "planResult", sp);

		// visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner) planner, p);

	}

	public void PolicyIterationExample(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}

		OOMDPPlanner planner = new PolicyIteration(domain, rf, tf, 0.99,
				hashingFactory, 0.0001, 100, 100);
		planner.planFromState(initialState);

		// create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner) planner);

		// record the plan results to a file
		p.evaluateBehavior(initialState, rf, tf).writeToFile(
				outputPath + "planResult", sp);

		// visualize the value function and policy
		this.valueFunctionVisualize((QComputablePlanner) planner, p);

	}

	public void QLearningExample(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}
		
		// uncomment the learning strategy you want to use (and comment-out the rest)
		// Q-learning with 0.1 epsilon-greedy policy
		// discount= 0.99; initialQ=0.0; learning rate=0.9
		 //LearningAgent agent = new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		
		//  Q-learning with Boltzmann policy
		//LearningAgent agent = new MyBoltzmannQLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		
		//  Q-learning with GreedyQ policy
		//LearningAgent agent = new MyGreedyQLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		 
		// Q-learning using epsilon-greedy, with tweaked epsilon
		 LearningAgent agent = new MyEpsilonGreedyQLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		
		// run learning for 100 episodes
		for (int i = 0; i < 30; i++) {
			EpisodeAnalysis ea = agent.runLearningEpisodeFrom(initialState);
			ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
			System.out.println(i + ": " + ea.numTimeSteps());
		}

	}

	public void valueFunctionVisualize(QComputablePlanner planner, Policy p) {
		List<State> allStates = StateReachability.getReachableStates(
				initialState, (SADomain) domain, hashingFactory);
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTY);

		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT,
				GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH,
				new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH,
				new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST,
				new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST,
				new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
				allStates, svp, planner);
		gui.setSpp(spp);
		gui.setPolicy(p);
		gui.setBgColor(Color.GRAY);
		gui.initGUI();
	}

	public void experimenterAndPlotter() {

		// custom reward function for more interesting results
		final RewardFunction rf = new GoalBasedRF(this.goalCondition, 5., -0.1);

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */

		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			@Override
			public String getAgentName() {
				return "Q-learning";
			}

			@Override
			public LearningAgent generateAgent() {
				return new QLearning(domain, rf, tf, 0.99, hashingFactory, 0.3,
						0.1);
			}
		};

		LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

			@Override
			public String getAgentName() {
				return "SARSA";
			}

			@Override
			public LearningAgent generateAgent() {
				return new SarsaLam(domain, rf, tf, 0.99, hashingFactory, 0.0,
						0.1, 1.);
			}
		};

		StateGenerator sg = new ConstantStateGenerator(this.initialState);

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
				(SADomain) this.domain, rf, sg, 10, 100, qLearningFactory,
				sarsaLearningFactory);

		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOSTRECENTANDAVERAGE,
				PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
				PerformanceMetric.AVERAGEEPISODEREWARD);

		exp.startExperiment();

		exp.writeStepAndEpisodeDataToCSV("expData");

	}

}