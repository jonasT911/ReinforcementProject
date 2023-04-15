import torch
import random 
import numpy as np
from collections import deque

import sys
sys.path.append("..") #Import parent folder
from vector import Vector2
from constants import *
from run import GameController

from MLfiles.model import Linear_QNet, QTrainer
from MLfiles.helper import plot

from constants import *

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001 #Was .001



class Agent:
    
    def __init__(self):
        self.n_games=0
        self.epsilon = 0 #controls randomness      
        self.gamma=0.01 #discount rate must be smaller than 1
        self.memory = deque(maxlen = MAX_MEMORY) #popleft()
        self.model = Linear_QNet(520,3048,4) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma) 
        self.holdRandom = 0
        self.Random_move = [0,0,0,0]
        self.previousLocation= Vector2(0,0)
      
    def penalizeToLastTurn(self,penalty):
        print("Penalizing")
        i=len(self.memory)-1
        openDirections=0
        action=[1,0,0,0]
        reward=10
        while (i>0 and( action[0] !=1 )and reward<=0):
             state, action, reward, next_state,done=self.memory[i]
             reward=reward+penalty
             self.memory[i]=(state, action, reward, next_state, done)
             openDirections=int(state[2])+int(state[3])+int(state[4])+int(state[5])
             i=i-1
             
    def rewardUntilLastPenalty(self,extraReward):
        print("Rewarding")
        i=len(self.memory)-1
        if(i<0):
            return
        state, action, old_reward, next_state,done=self.memory[i]
        sameAction=state[6:10]
        openDirections=0
        
        while (i>0 and old_reward>=0):
             state, action, old_reward, next_state,done=self.memory[i]
             openDirections=int(state[2])+int(state[3])+int(state[4])+int(state[5])
             if(old_reward>=0):
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)
             
             i=i-1
        print("ACTION WAS "+str(len(self.memory)-1-i))
             
    def starve(self,extraReward):
        print("Starving")
        i=len(self.memory)-1
        state, action, old_reward, next_state,done=self.memory[i]
        sameAction=action
        openDirections=0
        
        while (i>0 and old_reward<=0 and len(self.memory)-i<30):
             state, action, old_reward, next_state,done=self.memory[i]
             
             if(old_reward<=0):
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)
             
             i=i-1
             
             
    def evaluateWholeRun(self,extraReward):
        print("Rewarding")
        i=len(self.memory)-2
        state, action, old_reward, next_state,done=self.memory[i]
        done = False
        print("Run rewards are ")
        print(old_reward)
        while (i>=0 and not done):
             state, action, old_reward, next_state,done=self.memory[i]
           
             if(not done):
                 print(old_reward)
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)          
             i=i-1
        dist=len(self.memory)-i
        
        print("MEMORY = "+str(len(self.memory)))
        print("RUn was length "+str(dist))
             
    def get_state(self, game):
    
      
#State includes pacman location, ghost location,ghost direction, array of uneaten pellets, available turns.
	#Maybe power pellets, ghosts state(VERY IMPORTANT), fruit, wall location/straight path to ghost? (This should be covered by direction
	#For my paper's purpose: ghost behavior.
	
        pacLoc = game.pacman.position
        print("Pac location is ("+str(pacLoc.x)+", "+str(pacLoc.y)+")")
        
        blink=game.ghosts.blinky
        pink=game.ghosts.pinky
        ink=game.ghosts.inky
        clyde=game.ghosts.clyde
        pellets=game.pellets.pelletList
        
        dir_l = game.pacman.pointing == LEFT
        dir_r = game.pacman.pointing == RIGHT
        dir_u = game.pacman.pointing == UP
        dir_d = game.pacman.pointing == DOWN
        stopped= game.pacman.direction==STOP
        
        openLeft=game.pacman.getNewTarget(game.pacman.drivingControl([0,0,1,0])) is not game.pacman.node
        openRight=game.pacman.getNewTarget(game.pacman.drivingControl([0,1,0,0])) is not game.pacman.node
        
        state = [
		pacLoc.x,
		pacLoc.y,
		  # Move direction

		#Open positions for pac man 
		self.previousLocation.x,
		self.previousLocation.y,
		openLeft,
		
		openRight,
		dir_u,
		dir_d,	
		dir_l,
		dir_r,
		
		
		
	#	Ghost Positions
		blink.position.x,
		blink.position.y,
		blink.mode.current == FREIGHT,
		blink.mode.current == SPAWN,
		
		pink.position.x,
		pink.position.y,
		pink.mode.current == FREIGHT,
		pink.mode.current == SPAWN,
	#NOTHING CAN BE PUT IN FRONT OF THIS!
	#	
	#	ink.position.x,
	#	ink.position.y,
	#	ink.mode.current == FREIGHT,
	#	ink.mode.current == SPAWN,
		
	#	clyde.position.x,
	#	clyde.position.y,
	#	clyde.mode.current == FREIGHT,
	#	clyde.mode.current == SPAWN,
		#Ghost Direction 
		
	        stopped,
         
              
        ]
        
        self.previousLocation=pacLoc
        
        #Add the pellets to the list, and show if they are active.
        blinkDir=[0,0,0,0,0]
        blinkDir[2+blink.direction]=1
        state.extend(blinkDir)
        
        ink_Dir=[0,0,0,0,0]
        ink_Dir[2+ink.direction]=1
        #state.extend(ink_Dir)
        
        pinkDir=[0,0,0,0,0]
        pinkDir[2+pink.direction]=1
        state.extend(pinkDir)
        
        clydeDir=[0,0,0,0,0]
        clydeDir[2+clyde.direction]=1
        #state.extend(clydeDir)
        
        nearest,nearDistance=game.pacman.nearestPellet(pellets)
        state.extend([nearest.position.x,nearest.position.y,nearDistance])
        
        pelletLocations=[]
        for i in pellets:
            if (not i.eaten):
               pelletLocations.extend([i.position.x,i.position.y])#Need better boolean. Visible is for special effects
            else:
               pelletLocations.extend([-1,-1])#Need better boolean. Visible is for special effects
        state.extend(pelletLocations)
      
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))#If this exceeds max mem then pop left
    
    def train_long_memory(self):
	
        #Can be changed later
        print("Long memory")
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states,dones = zip (*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
    def get_action(self,state):
        #random moves: tradeoff exploitation/exploration
        self.epsilon = 20 - self.n_games
        if (self.epsilon<0):
            self.epsilon=10#Always ensures a bit of randomness
        final_move = [0,0,0,0]
        if random.randint(0,20)<self.epsilon:
            move =random.randint(0,2) #Dropped to 2 while I can not reverse
            final_move[move] = 1
        else:
            state0 =torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            print("Prediction: "+str(prediction))
            move = torch.argmax(prediction).item() #Change the function to return one of four directions.
            final_move[move]=1
        return final_move
        
def train ():
    static=0
    plot_scores=[]
    plot_mean_scores= []
    total_score=0
    record = 0
    agent = Agent()
    game = GameController(True)
    game.startGame()
    game.MachineLearning=True
    game.lives=1
    mean_score=0
    starving =0
    
    lastPellet,lastPelletDist=game.pacman.nearestPellet(game.pellets.pelletList)

    while True:
        print("       ")
        #get old state
        state_old = agent.get_state(game)
        #print(state_old)
        
        
        
        #get move
        final_move = agent.get_action(state_old)
        
        #perform move and get new state
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        reward=reward*10
        print(final_move)
        print("{"+str(state_old[2])+str(state_old[3])+str(state_old[4])+str(state_old[5])+"}")

        #Penalize getting close to ghosts
        d = Vector2(state_old[0], state_old[1]) - Vector2(state_old[10],state_old[11])
        oldDist=d.magnitudeSquared()
        d = Vector2(state_new[0], state_new[1]) - Vector2(state_new[10],state_new[11])
        distance= d.magnitudeSquared()
    
        if( distance<1000  and state_old[12]==0 and  state_old[13]==0 ):
            if (oldDist>distance ):
                reward=reward-0
            else:
                reward=reward+0
  
        d = Vector2(state_old[0], state_old[1]) - Vector2(state_old[14],state_old[15])
        oldDist=d.magnitudeSquared()
        d = Vector2(state_new[0], state_new[1]) - Vector2(state_new[14],state_new[15])
        distance= d.magnitudeSquared()
        if(oldDist>distance and distance<1000 and state_old[16]==0 and  state_old[17]==0 ):
            if (oldDist>distance ):
                reward=reward-0
            else:
                reward=reward+0
                

     
        if (state_new[0]==state_old[0] and state_new[1]==state_old[1]):
            if(static>5):
                print("Standing Still")
                #reward=reward-1#Standing still penalty
            else:
                static +=1
        else:
            static=0

        if(reward>0):           
            agent.rewardUntilLastPenalty(reward)#Propagates reward until a different action could have been taken.
            starving=0
        else:
            starving+=1
            if(starving>5):
        #        print("Penalty")
                reward=reward-0
                #agent.penalizeToLastTurn(-1)
        
        
        nearest,nearDistance=game.pacman.nearestPellet(game.pellets.pelletList)
        if(nearest==lastPellet):
            if (lastPelletDist>nearDistance):
                reward=reward
            else:
                reward=reward-0
        lastPelletDist=nearDistance
        lastPellet=nearest    
        print("Reward is "+str(reward))   
        
     
        agent.remember(state_old,final_move,reward,state_new,done)
        
        #Game ends
        if done or starving>1020:
            runScore=int((game.score -mean_score)/10)
            print("run Score is "+str(runScore))
            #agent.evaluateWholeRun(runScore)
            #train long memory, plot result
            
           
            
            agent.n_games+=1
            #Uncomment this later
            agent.train_long_memory()
            
            if score>record :
                record=score
                agent.model.save()
            print('Game',agent.n_games, 'Score',score,'Record:',record)
            #Plot
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            
            game.restartGame()
            game.lives=1
            

if __name__== '__main__':
	print("Begin ML Pac-Man")
	train()
