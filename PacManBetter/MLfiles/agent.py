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

MAX_MEMORY = 1000000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        self.n_games=0
        self.epsilon = 0 #controls randomness      
        self.gamma=0.9 #discount rate must be smaller than 1
        self.memory = deque(maxlen = MAX_MEMORY) #popleft()
        self.model = Linear_QNet(518,1024,4) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma) 
      
    def penalizeToLastTurn(self,penalty):
        print("Penalizing")
        i=len(self.memory)-1
        openDirections=0
        while (i>0 and( len(self.memory)-i<3 )):
             state, action, reward, next_state,done=self.memory[i]
             reward=reward+penalty
             self.memory[i]=(state, action, reward, next_state, done)
             openDirections=int(state[2])+int(state[3])+int(state[4])+int(state[5])
             i=i-1
             
    def rewardUntilLastPenalty(self,extraReward):
        print("Rewarding")
        i=len(self.memory)-1
        openDirections=0
        old_reward=0
        while (i>0 and openDirections<3 and old_reward>=0):
             state, action, old_reward, next_state,done=self.memory[i]
             openDirections=int(state[2])+int(state[3])+int(state[4])+int(state[5])
             if(old_reward>0):
                 reward=old_reward+extraReward
                 self.memory[i]=(state, action, reward, next_state, done)
             
             i=i-1
    def get_state(self, game):
    
      
#State includes pacman location, ghost location,ghost direction, array of uneaten pellets, available turns.
	#Maybe power pellets, ghosts state(VERY IMPORTANT), fruit, wall location/straight path to ghost? (This should be covered by direction
	#For my paper's purpose: ghost behavior.
	
        pacLoc = game.pacman.position

        
        blink=game.ghosts.blinky
        pink=game.ghosts.pinky
        ink=game.ghosts.inky
        clyde=game.ghosts.clyde
        pellets=game.pellets.pelletList
        

        
        state = [
		pacLoc.x,
		pacLoc.y,
		  # Move direction

		#Open positions for pac man 
		#NOTHING CAN BE PUT IN FRONT OF THESE!
		game.pacman.getNewTarget(LEFT) is not game.pacman.node,
		game.pacman.getNewTarget(RIGHT)is not game.pacman.node,
		game.pacman.getNewTarget(UP) is not game.pacman.node,
		game.pacman.getNewTarget(DOWN)is not game.pacman.node,
		
		
		
	#	Ghost Positions
		blink.position.x,
		blink.position.y,
		blink.mode.current == FREIGHT,
		blink.mode.current == SPAWN,
		
	#	pink.position.x,
	#	pink.position.y,
	#	pink.mode.current == FREIGHT,
	#	pink.mode.current == SPAWN,
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
		#These need to be changed so that left and right are not twice up and down.
	
		

        ]
        #Add the pellets to the list, and show if they are active.
        blinkDir=[0,0,0,0,0]
        blinkDir[2+blink.direction]=1
        state.extend(blinkDir)
        
        ink_Dir=[0,0,0,0,0]
        ink_Dir[2+ink.direction]=1
        state.extend(ink_Dir)
        
        pinkDir=[0,0,0,0,0]
        pinkDir[2+pink.direction]=1
        state.extend(pinkDir)
        
        clydeDir=[0,0,0,0,0]
        clydeDir[2+clyde.direction]=1
        state.extend(clydeDir)
        
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
        if (self.epsilon<1):
            self.epsilon=1#Always ensures a bit of randomness
        final_move = [0,0,0,0]
        if random.randint(0,100)<self.epsilon:
            move =random.randint(0,3)
            final_move[move] = 1
        else:
            state0 =torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() #Change the function to return one of four directions.
            final_move[move]=1
        return final_move
        
        
def train ():
    static=False
    plot_scores=[]
    plot_mean_scores= []
    total_score=0
    record = 0
    agent = Agent()
    game = GameController(True)
    game.startGame()
    game.MachineLearning=True
    game.lives=1
    

    while True:
        #get old state
        state_old = agent.get_state(game)
        #print(state_old)
        
        d = Vector2(state_old[0], state_old[1]) - Vector2(state_old[6],state_old[7])
        oldDist=d.magnitudeSquared()
        
        #get move
        final_move = agent.get_action(state_old)
        
        #perform move and get new state
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        ##Penalize getting close
        d = Vector2(state_new[0], state_new[1]) - Vector2(state_new[6],state_new[7])
  
        distance= d.magnitudeSquared()
        print(distance)
        if(oldDist>distance and distance<1000):
            reward=reward-100
        #if(reward<-9):
        #    print("Penalty")
        #    agent.penalizeToLastTurn(reward)
     
        if (state_new[0]==state_old[0] and state_new[1]==state_old[1]):
            if(static):
                reward=reward-40#Standing still penalty
            else:
                static =True
        else:
            static=False
        
        
        if(reward>0):           
            agent.rewardUntilLastPenalty(reward)#Propagates reward until a different action could have been taken.
        print("Reward is "+str(reward))   
        
        
        #train short memory
        openDirections=int(state_old[2])+int(state_old[3])+int(state_old[4])+int(state_old[5])
       
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
            #remember
        if(openDirections>1):#Useless if 1. Was >2
            agent.remember(state_old,final_move,reward,state_new,done)
        
        if done:
            #train long memory, plot result
            game.restartGame()
            game.lives=1
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
            
            

if __name__== '__main__':
	print("Begin ML Pac-Man")
	train()
