# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() #이동한 위치의 좌표
        newFood = successorGameState.getFood() #맵 전체에 있는 먹이의 위치 T/F
        newGhostStates = successorGameState.getGhostStates() #ghost들의 위치. [(x, y), (x2, y2) ... ]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #각 ghost들의 scared time 잔여량 [ghost1의 잔여량, ghost 2의 잔여량 ...]

        "*** YOUR CODE HERE ***"
        score = 0
        #Meet Ghost
        GhostPositions = [Ghost.getPosition() for Ghost in newGhostStates]
        index = 0
        for GhostPos in GhostPositions:
            if util.manhattanDistance(GhostPos, newPos) <= 1:
                if newScaredTimes[index] > 0: return 9999
                return -9999
            index = index +1
        #find nearest food'
        curFood = currentGameState.getFood()
        near = 9999
        for i in range(curFood.width):
            for j in range(curFood.height):
                if curFood[i][j] and util.manhattanDistance((i, j), newPos) < near:
                    near = util.manhattanDistance((i, j), newPos)
                    if near == 0: break
        return -near

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def pacmanTurn(self, gameState, curDepth):
        actions = gameState.getLegalActions(0)
        v = -99999
        tmp = v
        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            if succ.isWin() or succ.isLose() or curDepth == self.depth:
                tmp = self.evaluationFunction(succ)
            else:
                tmp = self.ghostTurn(succ, curDepth + 1, 1)
            v = max(v, tmp)
        return v

    def ghostTurn(self, gameState, curDepth, agentID):
        ghostEndNum = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(agentID)
        actions_pac = gameState.getLegalActions(0)
        v = 99999
        tmp = v
        for action in actions:
            succ = gameState.generateSuccessor(agentID, action)
            if succ.isLose() or succ.isWin():
                tmp = self.evaluationFunction(succ)
            else:
                if agentID == ghostEndNum:
                    #end point
                    if curDepth == self.depth: tmp = self.evaluationFunction(succ)
                    else: tmp = self.pacmanTurn(succ, curDepth)
                else:
                    tmp = self.ghostTurn(succ, curDepth, agentID + 1)
            v = min(v, tmp)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        bestAction = actions[0]
        v = -99999
        for action in actions:
            succ = gameState.generateSuccessor(0, action)
            if succ.isWin() or succ.isLose(): tmp = self.evaluationFunction(succ)
            else:
                tmp = self.ghostTurn(succ, 1, 1)
            if tmp > v:
                v = tmp
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
