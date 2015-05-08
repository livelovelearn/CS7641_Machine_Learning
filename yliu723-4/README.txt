I. Tools:
The Burlap (Brown-UMBC Reinforcement Learning and Planning) java library is used for this assignment.   

The installation instructions can be found at:
http://burlap.cs.brown.edu/

The Burlap source code can also be found at Github:
https://github.com/jmacglashan/burlap


II. Markov Decision Processes:
The MDPs used for analysis are Four-Room (11x11) and Random-Maze (25x25).Both are modified from Burlap gridWorld domain.

III. Code file:
1, BacisBehavior_FourRoom.java (FourRoom MDP analyzed with VI, PI and Q-learning) 
2, BacisBehavior_Maze.java (RandomMaze MDP analyzed with VI, PI and Q-learning) 
3, MazeDomain.java (modified gridWorld domain for random maze layout)
4, MyEpsilonGreedyQLearning.java (modified QLearing class using EpsilonGreedy strategy and to tweak parameter epsilon)
5, MyBlotzmannQLearning.java (modified QLearning class using Boltzmann strategy and to tweak parameter temperature)
6, MyGreedyQLearning.java (modified QLearning class using greedy strategy)







	  
